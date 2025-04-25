# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Engines to handle the work when executing a TVM compiled NN model."""
import os
import numpy as np
import tvm
from tvm import nd, runtime, relax


class ExecutionEngine(tvm.Object):
    """The engine that is used to execute the NN models compiled for VM or the graph executor.

    Attributes
    ----------
    type_key : str
        The string that indicates the underlying object type. "VmExecutionEngine" is used to execute
        the NN models compiled for VM. "GraphExecutionEngine" is used to execute the NN models
        compiled for the graph executor.

    executor : Union[vm.VirtualMachine, VirtualMachineProfiler, GraphModule, GraphModuleDebug]
        The contained object of the official executor API. It is an object of Python class
        "vm.VirtualMachine" for the NN models compiled for VM, or an object of Python class
        "graph_executor.GraphModule" for the NN models compiled for the graph executor, or an object
        of the corresponding version that with profiling ability. All official executor APIs can be
        called through this object, for example, "benchmark", "profile".
    """

    def __init__(
        self, compiled_model, rpc_sess=None, devices=None, with_profile=False
    ):  # pylint: disable=unused-argument
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
            The devices on which to execute the compiled NN model. If it isn't set, rpc_sess.cpu(0)
            will be used if the "rpc_sess" is set, and tvm.cpu(0) will be used if the "rpc_sess"
            isn't set.

        with_profile : bool
            Whether select the execution engine that with profiling ability or not. If it is True,
            the underlying class of execution engine will use "VirtualMachineDebug" or
            "GraphExecutorDebug" instead of "VirtualMachine" or "GraphExecutor".
        """
        if rpc_sess:
            rpc_sess.upload(compiled_model)
            compiled_model = rpc_sess.load_module(os.path.basename(compiled_model))
        elif isinstance(compiled_model, str):
            compiled_model = runtime.load_module(compiled_model)

        if devices is None:
            devices = [rpc_sess.cpu(0)] if rpc_sess else [tvm.cpu(0)]
        elif not isinstance(devices, (list, tuple)):
            devices = [devices]

        self.executor = relax.VirtualMachine(compiled_model, devices)
        self.devices = devices

    def _convert2ndarray(self, args):
        nd_arrs = []
        for arg in args:
            if isinstance(arg, nd.NDArray):
                nd_arrs.append(arg)
            elif isinstance(arg, np.ndarray):
                # todo
                # dev = runtime._ffi_api.ExecutionEngine_GetInputDevice(self, i)
                dev = self.devices[0]
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
        self.executor.set_input("main", *args)

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

        self.executor.invoke_stateful("main")
        outputs = self.executor.get_outputs("main")
        return outputs if isinstance(outputs, tuple) else (outputs,)

    def profile(self, *args):
        """The API that is used to profile the compiled NN model.

        Parameters
        ----------
        *args : Tuple of tvm.nd.NDArray or numpy.ndarray
            The tuple of all real input data. If it is not empty, the API set_inputs will be
            invocated with its value.
        """
        return self.executor.profile("main", args)
