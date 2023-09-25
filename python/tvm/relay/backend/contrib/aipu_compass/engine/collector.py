# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""Collect calibration dataset through forward propagation."""
import os
import numpy as np
import tvm
from tvm import relay
from .engine import PurePythonForwardEngine, FunctionData
from ..aipu_builder import OptForward
from ..codegen import CodeGenAipuCompass
from ..config import AipuCompassFunctionConfig
from .. import utils


class _FunctionData(FunctionData):
    def __init__(self, rly_func):
        super().__init__(rly_func)
        self.batched_in_args = []

    def collect_in_args(self, in_args):
        """Get input data for forward."""

        def _unpack_tuple_args(args):
            unpack_args = []
            for arg in args:
                if isinstance(arg, tuple):
                    unpack_args += list(arg)
                else:
                    unpack_args.append(arg)
            return unpack_args

        in_args = _unpack_tuple_args(in_args)
        if len(self.batched_in_args) == 0:
            self.batched_in_args = [x.numpy() for x in in_args]
            return

        assert len(self.batched_in_args) == len(in_args)
        for i, batched_arg in enumerate(self.batched_in_args):
            self.batched_in_args[i] = np.concatenate((batched_arg, in_args[i].numpy()))


def _create_opt_executor(func):
    # Get the Compass IR from Relay IR and write them to disk.
    cfg = AipuCompassFunctionConfig(func.attrs.global_symbol)
    txt_path, bin_path = cfg.compass_ir_path
    CodeGenAipuCompass().gen2file(func, txt_path, bin_path)
    return OptForward(txt_path, bin_path)


def _create_relay_vm_executor(func):
    # Create a new IRModule from the AIPU Compass function with removing all
    # function attributes.
    ir_mod = tvm.IRModule.from_expr(func.body)

    # Do any optimizations needed, e.g., convert the layout according to CPU
    # requirement.
    passes = [
        relay.transform.ConvertLayout(utils.X86_DESIRED_LAYOUTS),
        relay.transform.FoldConstant(),
    ]
    with tvm.transform.PassContext(opt_level=3):
        ir_mod = tvm.transform.Sequential(passes)(ir_mod)

    # Compile the new IRModule for running on CPU.
    return relay.create_executor(kind="vm", mod=ir_mod).evaluate()


class AipuCollector(PurePythonForwardEngine):
    """Running AIPU Compass function through forward engine, and collecting
    calibration dataset of each AIPU Compass function."""

    def __init__(self, forward_engine):
        super().__init__()
        self._forward_engine = forward_engine

    def run(self, func_data, args):
        """Responsible for executing AIPU Compass function during relay run the
        whole compiled model."""
        func_data.collect_in_args(args)
        if self._forward_engine == "relay_vm":
            return func_data.executor(*args)
        assert self._forward_engine == "opt"
        return func_data.executor.forward([x.numpy() for x in args], True)

    def create_func_data(self, func):
        """Responsible for creating object used to store data of the Relay
        function."""
        func_data = _FunctionData(func)
        if self._forward_engine == "relay_vm":
            func_data.executor = _create_relay_vm_executor(func)
        else:
            assert self._forward_engine == "opt"
            func_data.executor = _create_opt_executor(func)

        return func_data

    def finish(self, batch_size):
        """Save calibration dataset of subgraphs.
        batch_size: batch size of entrance dataset.
            Compare subgraphs' shape[0] and batch size to check if without
            batch dim. Need expand batch dim in subgraphs' dataset if true.
        """
        for func_name, func_data in self._name2func_data.items():
            without_batch_dim = False
            for i, batched_arg in enumerate(func_data.batched_in_args):
                shape = batched_arg.shape
                if shape[0] > batch_size:
                    new_shape = [batch_size, -1] + list(shape[1:])
                    func_data.batched_in_args[i] = batched_arg.reshape(new_shape)
                    without_batch_dim = True
            cfg = AipuCompassFunctionConfig(func_name)
            os.makedirs(os.path.dirname(cfg.calibration_data), exist_ok=True)
            np.savez(cfg.calibration_data, *func_data.batched_in_args)
            if without_batch_dim:
                cfg.gen_optimizer_config_file(extra_cfg={"without_batch_dim": "true"})
