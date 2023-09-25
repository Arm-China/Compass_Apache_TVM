# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""Provide a mechanism to process AIPU Compass function flexibly."""
import numpy as np
import tvm
from tvm import ir, runtime
from ..utils import convert_to_tuple


class AipuForwardEngine:
    """Base class of context managers that determine how to build & run AIPU
    Compass function."""

    current = None

    def __enter__(self):
        self._old_engine = AipuForwardEngine.current
        AipuForwardEngine.current = self
        return self

    def __exit__(self, ptype, value, trace):
        AipuForwardEngine.current = self._old_engine

    def pre_build(self, func):
        """Responsible for processing AIPU Compass function before Relay build
        the whole model."""
        return func

    def build(self, func):
        """Responsible for processing AIPU Compass function during Relay build
        the whole model."""
        raise NotImplementedError


class FunctionData:
    """Base class of structure that store data of Relay function."""

    def __init__(self, rly_func):
        self._in_param_cnt = 0
        self._in_param_num = []
        for param in rly_func.params:
            param_type = param.checked_type
            num = len(param_type.fields) if isinstance(param_type, ir.TupleType) else 1
            self._in_param_cnt += num
            self._in_param_num.append(num)
        ret_type = rly_func.checked_type.ret_type
        self._out_param_cnt = len(ret_type.fields) if isinstance(ret_type, ir.TupleType) else 1
        self.name = rly_func.attrs.global_symbol
        self.executor = None

    def split_args(self, args):
        assert len(args) == self._in_param_cnt + self._out_param_cnt
        in_args = []
        begin = 0
        for num in self._in_param_num:
            arg = args[begin : num + begin]
            in_args += arg if num == 1 else [arg]
            begin += num
        return (in_args, args[-(self._out_param_cnt) :])


class PurePythonForwardEngine(AipuForwardEngine):
    """Base class of AIPU forward engines that build & run AIPU Compass function
    in pure Python code."""

    def __init__(self):
        super().__init__()
        self._name2func_data = dict()

    def run(self, func_data, args):
        """Responsible for executing AIPU Compass function during Relay run the
        whole compiled model."""
        raise NotImplementedError

    def create_func_data(self, func):
        """Responsible for creating object used to store data of the Relay
        function."""
        raise NotImplementedError

    def build(self, func):
        """Build the AIPU Compass function for running it through Python code
        during Relay build the whole model."""
        func_name = func.attrs.global_symbol
        self._name2func_data[func_name] = self.create_func_data(func)

        # The decorator is just used to make the function "wrapper" become a
        # packed function.
        @tvm.register_func("PurePythonForwardEngine", override=True)
        def wrapper(*args):
            func_data = self._name2func_data[func_name]
            in_args, out_args = func_data.split_args(args)

            ret_values = convert_to_tuple(self.run(func_data, in_args))

            assert len(out_args) == len(ret_values)
            for out_arg, ret_value in zip(out_args, ret_values):
                assert isinstance(out_arg, runtime.NDArray)
                assert isinstance(ret_value, (runtime.NDArray, np.ndarray))
                if out_arg.shape != ret_value.shape:
                    assert isinstance(ret_value, np.ndarray)
                    ret_value = ret_value.reshape(out_arg.shape)
                out_arg.copyfrom(ret_value)

        # Return a PackedFuncModule to let TVM runtime call back later.
        return runtime._ffi_api.PackedFuncModuleCreate(func_name, wrapper)
