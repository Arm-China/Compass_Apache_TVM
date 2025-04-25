# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Execute AIPU compass function through AIPU GT executor."""
import os
import numpy as np
from tvm import nd
from tvm.aipu.utils import check_call_aipu_tool
from AIPUBuilder.executor import GtForward
from .engine import PurePythonForwardEngine, FunctionData
from ..codegen import CodeGenAipuCompass
from ..config import AipuCompassConfig, AipuCompassFunctionConfig
from ..utils import relative_symlink_in_dir


class AIPUGtForwardEngine(PurePythonForwardEngine):
    """Build AIPU Compass function for running through AIPU GT executor."""

    def run(self, func_data, args):
        """Responsible for executing AIPU Compass function during Relax run the
        whole compiled model."""
        np_args = [x.numpy() for x in args]
        ret = func_data.executor.forward(np_args)

        # Dump all quantized input and output data for debugging with tool like "aipuexe" if needed.
        if AipuCompassConfig.get().runtime["dump"] == "true":
            cfg = AipuCompassFunctionConfig(func_data.name)
            os.makedirs(cfg.gbuilder_work_dir, exist_ok=True)

            for i, np_arg in enumerate(np_args):
                np_arg.tofile(f"{cfg.gbuilder_work_dir}/input{i}_{np_arg.dtype}.bin")
            for i, np_ret in enumerate(ret):
                np_ret.tofile(f"{cfg.gbuilder_work_dir}/output{i}_{np_ret.dtype}.bin")
        return ret

    def pre_build(self, func):
        """Build the AIPU Compass function to run on AIPU GT executor before
        Relax build the whole model."""
        # Get the Compass IR from Relax IR.
        cfg = AipuCompassFunctionConfig(func.attrs.global_symbol)
        CodeGenAipuCompass().gen2file(func, *cfg.compass_ir_path)
        ir_path = cfg.compass_ir_path

        # Simplify Compass float IR through "aipugsim" if needed.
        if cfg.use_gsim_float:
            # Create symbolic links that point to Compass IR in AIPU gsim_float directory.
            relative_symlink_in_dir(cfg.compass_ir_path, cfg.gsim_float_work_dir)
            check_call_aipu_tool(cfg.gsim_cmd("float"), cfg.gsim_float_work_dir)
            ir_path = cfg.gsim_float_ir_path

        # Get the quantized Compass IR through AIPU Optimizer.
        # Create symbolic links that point to Compass IR in AIPU Optimizer directory.
        float_ir_path = cfg.gsim_float_ir_path if cfg.use_gsim_float else cfg.compass_ir_path
        relative_symlink_in_dir(float_ir_path, cfg.optimizer_work_dir)
        check_call_aipu_tool(cfg.optimizer_cmd, cfg.optimizer_work_dir)
        ir_path = cfg.quant_compass_ir_path

        # Read back the Compass IR.
        txt_path, bin_path = ir_path
        ir_txt = open(txt_path, "r", encoding="utf-8").read()
        ir_bin = np.fromfile(bin_path, dtype="uint8")

        try:
            # Try to obtain the NDArray object without memory copy.
            ir_bin = nd.from_dlpack(ir_bin)
        except:  # pylint: disable=bare-except
            ir_bin = nd.array(ir_bin)

        # Embed the pre-build result into Relax IR through attribute of function.
        new_attrs = {"compass.pre_build.ir_txt": ir_txt, "compass.pre_build.ir_bin": ir_bin}
        return func.with_attr(new_attrs)

    def create_func_data(self, func):
        """Responsible for creating object used to store data of the Relax
        function with the pre-build result during Relax build the whole model."""
        func_data = FunctionData(func)

        # Write the Compass IR to disk for AIPU GT executor forward interface.
        cfg = AipuCompassFunctionConfig(func_data.name)
        txt_path, bin_path = cfg.quant_compass_ir_path

        # Get the pre-build result back from Relax IR.
        # There is a bug in Relax parser, it won't resume the escaped characters.
        ir_txt = func.attrs["compass.pre_build.ir_txt"].encode("utf-8").decode("unicode_escape")
        ir_bin = func.attrs["compass.pre_build.ir_bin"].numpy()

        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        open(txt_path, "w", encoding="utf-8").write(ir_txt)
        os.makedirs(os.path.dirname(bin_path), exist_ok=True)
        ir_bin.tofile(bin_path)
        target = AipuCompassConfig.get().gbuilder["target"]
        # Create the AIPU GT executor forward instance.
        func_data.executor = GtForward(txt_path, bin_path, target=target)
        return func_data
