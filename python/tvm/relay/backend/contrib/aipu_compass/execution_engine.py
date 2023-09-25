# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""Engines to handle the work when executing a TVM compiled NN model."""
import os
import numpy as np
import tvm
from tvm import contrib, nd, runtime


# The type key string here must be identical with the value of C++ variable
# "VmExecutionEngineObj::_type_key" and "GraphExecutionEngineObj::_type_key".
@tvm.register_object("VmExecutionEngine")
@tvm.register_object("GraphExecutionEngine")
class ExecutionEngine(tvm.Object):
    """The engine that is used to execute the NN models compiled for VM or the graph executor.

    Attributes
    ----------
    type_key : {"VmExecutionEngine", "GraphExecutionEngine"}
        The string that indicates the underlying object type. VmExecutionEngine is used to execute
        the NN models compiled for VM. GraphExecutionEngine is used to execute the NN models
        compiled for the graph executor.

    executor : tvm.runtime.vm.VirtualMachine or tvm.contrib.graph_executor.GraphModule
        The contained object of the official executor API. It is an object of Python class
        tvm.runtime.vm.VirtualMachine for the NN models compiled for VM and an object of Python
        class tvm.contrib.graph_executor.GraphModule for the NN models compiled for the graph
        executor. All official executor APIs can be called through this object, for example,
        benchmark.
    """

    def __init__(self, compiled_model, rpc_sess=None, devices=None):
        """The constructor of this class.

        Parameters
        ----------
        compiled_model : str or tvm.runtime.Module
            The value not only can be the deployed file path, but also can be the equivalent object
            in memory.

        rpc_sess : tvm.rpc.RPCSession
            The RPC session that is already connected to the RPC server. If it is set, the parameter
            compiled_model must be the deployed file path, and the file will be uploaded to the RPC
            server and loaded back automatically.

        devices : tvm._ffi.runtime_ctypes.Device or list of tvm._ffi.runtime_ctypes.Device
            The devices on which to execute the compiled NN model. If it is not set, rpc_sess.cpu(0)
            will be used if the rpc_sess is set, and tvm.cpu(0)will be used if the rpc_sess is not
            set.
        """
        if rpc_sess:
            rpc_sess.upload(compiled_model)
            compiled_model = rpc_sess.load_module(os.path.basename(compiled_model))

        if devices is None:
            devices = [rpc_sess.cpu(0)] if rpc_sess else [tvm.cpu(0)]
        elif not isinstance(devices, (list, tuple)):
            devices = [devices]

        # Structure "Device" isn't derived from "ObjectRef", i.e., multiple
        # "Device" can't be represented by "Array<Device>", so it can't be
        # passed by a single "TVMArgValue".
        self.__init_handle_by_constructor__(
            runtime._ffi_api.ExecutionEngine, compiled_model, *devices
        )

        self.type_key = runtime._ffi_api.ExecutionEngine_GetTypeKey(self)
        executor_mod = runtime._ffi_api.ExecutionEngine_GetExecutor(self)
        if self.type_key == "VmExecutionEngine":
            self.executor = runtime.vm.VirtualMachine(None, None, module=executor_mod)
        else:
            assert self.type_key == "GraphExecutionEngine"
            self.executor = contrib.graph_executor.GraphModule(executor_mod)

    def _convert2ndarray(self, args):
        nd_arrs = []
        for i, arg in enumerate(args):
            if isinstance(arg, nd.NDArray):
                nd_arrs.append(arg)
            elif isinstance(arg, np.ndarray):
                dev = runtime._ffi_api.ExecutionEngine_GetInputDevice(self, i)
                nd_arrs.append(nd.array(arg, dev))
            else:
                raise NotImplementedError(f'Convert "{type(arg)}" to NDArray.')
        return nd_arrs

    def set_inputs(self, *args):
        """The API that is used to set the real input data to the compiled NN model

        Parameters
        ----------
        *args : Tuple of tvm.nd.NDArray or numpy.ndarray
            The tuple of all real input data. If it is empty, nothing will be done. Any object of
            type numpy.ndarray will be converted to the object of type tvm.nd.NDArray first.
        """
        if not args:
            return
        nd_arrs = self._convert2ndarray(args)
        # Ensure the data type of arguments match that of parameters.
        for i, nd_arr in enumerate(nd_arrs):
            param_dtype = runtime._ffi_api.ExecutionEngine_GetEntryParamDataType(self, i)
            arg_dtype = nd_arr.dtype
            if param_dtype != "" and arg_dtype != param_dtype:  # pylint: disable=consider-using-in
                raise TypeError(
                    f"The type of the {i}th input mismatches that of the compiled model, "
                    f'expected: "{param_dtype}" vs real: "{arg_dtype}".'
                )
        runtime._ffi_api.ExecutionEngine_SetInputs(self, nd_arrs)

    def run(self, *args):
        """The API that is used to run the compiled NN model.

        Parameters
        ----------
        *args : Tuple of tvm.nd.NDArray or numpy.ndarray
            The tuple of all real input data. If it is not empty, the API set_inputs will be
            invocated with its value.

        Returns
        -------
        result : list of tvm.nd.NDArray
            The outputs of the compiled NN model for this execution. Its type is always list even
            though there is only one output.
        """
        if args:
            self.set_inputs(*args)
        return runtime._ffi_api.ExecutionEngine_Execute(self)
