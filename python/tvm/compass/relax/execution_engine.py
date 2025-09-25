# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Engines to handle the work when executing a TVM compiled NN model."""
import os
import shutil
import pathlib
import numpy as np
import tvm
from tvm import nd, runtime, relax
from ..runtime import _ffi_api


# The type key string here must be identical with the value of C++ variable
# "VmExecutionEngineObj::_type_key".
@tvm.register_object("compass.runtime.VmExecutionEngine")
class ExecutionEngine(tvm.Object):
    """The engine that is used to execute the NN models compiled for Relax VM.

    Attributes
    ----------
    executor : relax.VirtualMachine
        The contained object of the official executor API. It is an object of Python class
        "relax.VirtualMachine". All official executor APIs can be called through this object, for
        example, "time_evaluator", "profile".
    """

    def __init__(self, compiled_model, rpc_sess=None, devices=None, with_profile=False):
        """The constructor of this class.

        Parameters
        ----------
        compiled_model : str or tvm.runtime.Module
            The value not only can be the deployed directory path, but also can be the equivalent
            object in memory.

        rpc_sess : tvm.rpc.RPCSession
            The RPC session that is already connected to the RPC server. If it is set, the parameter
            compiled_model must be the deployed directory path, and the directory will be uploaded
            to the RPC server and loaded back automatically.

        devices : tvm.runtime.Device or list of tvm.runtime.Device
            The devices on which to execute the compiled NN model. If it isn't set, rpc_sess.cpu(0)
            will be used if the "rpc_sess" is set, and tvm.cpu(0) will be used if the "rpc_sess"
            isn't set.

        with_profile : bool
            Whether select the execution engine that with profiling ability or not. If it is True,
            the attribute "executor" will enable profiling.
        """
        if rpc_sess:
            root_dir, base_dir = os.path.split(compiled_model)
            archive_file = shutil.make_archive(f"{compiled_model}.cps", "tar", root_dir, base_dir)
            rpc_sess.upload(archive_file)
            compiled_model = rpc_sess.load_module(os.path.basename(archive_file))
        elif isinstance(compiled_model, (str, pathlib.Path)):
            compiled_model = runtime.load_module(compiled_model)

        if devices is None:
            devices = [rpc_sess.cpu(0)] if rpc_sess else [tvm.cpu(0)]
        elif not isinstance(devices, (list, tuple)):
            devices = [devices]

        # Structure "Device" isn't derived from "ObjectRef", i.e., multiple "Device" can't be
        # represented by "Array<Device>", so it can't be passed by a single "TVMArgValue".
        self.__init_handle_by_constructor__(
            _ffi_api.ExecutionEngine, compiled_model, with_profile, *devices
        )

        executor_mod = _ffi_api.ExecutionEngine_GetExecutor(self)
        self.executor = relax.VirtualMachine(None, None, module=executor_mod)
        self._with_profile = with_profile

    def _convert2ndarray(self, args):
        nd_arrs = []
        for i, arg in enumerate(args):
            if isinstance(arg, nd.NDArray):
                nd_arrs.append(arg)
            elif isinstance(arg, np.ndarray):
                dev = _ffi_api.ExecutionEngine_GetInputDevice(self, i)
                nd_arrs.append(nd.array(arg, dev))
            else:
                raise NotImplementedError(f'Convert "{type(arg)}" to NDArray.')
        return nd_arrs

    def set_inputs(self, *args):
        """The API that is used to set the real input data to the compiled NN model.

        Parameters
        ----------
        *args : Tuple of tvm.nd.NDArray or numpy.ndarray
            The tuple of all real input data. If it is empty, nothing will be done. Any object of
            type numpy.ndarray will be converted to the object of type tvm.nd.NDArray first.
        """
        if not args:
            return
        nd_arrs = self._convert2ndarray(args)
        _ffi_api.ExecutionEngine_SetInputs(self, nd_arrs)

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

        return _ffi_api.ExecutionEngine_Execute(self)

    def profile(self, *args):
        """The API that is used to profile the compiled NN model.

        Parameters
        ----------
        *args : Tuple of tvm.nd.NDArray or numpy.ndarray
            The tuple of all real input data. If it is not empty, the API set_inputs will be
            invocated with its value.

        Returns
        -------
        report: tvm.runtime.profiling.Report
            The formatted profiling result, showing per-op timing measurements.
        """
        assert self._with_profile, 'The execution engine must be created by "with_profile=True".'

        if args:
            self.set_inputs(*args)
        return self.executor.profile("main")
