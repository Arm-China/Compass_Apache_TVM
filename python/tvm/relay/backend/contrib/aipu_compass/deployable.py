# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Code to handle the work when deploying a TVM compiled NN model."""
from .config import AipuCompassConfig
from .execution_engine import ExecutionEngine


class Deployable:
    """The Relay build result for VM or the graph executor.
    This class should only be instanced by the compilation Python API."""

    def __init__(self, compiled_model):
        self._compiled_model = compiled_model

    def _check_exportable(self):
        if self._compiled_model.module._collect_from_import_tree(
            lambda x: x.type_key == "PackedFuncModule"
        ):
            raise RuntimeError(
                "Can't export the deployable file, because AIPU Compass engine is "
                'not set to "driver" when compiling the NN model.'
            )

    def export(self, *args, **kwargs):
        """The API for exporting the compiled NN model as a dynamic link file.

        All parameters will be passed to the underlying function export_library of
        Python class tvm.runtime.Module.
        For detailed explanation of each parameter, see the Apache TVM official document.
        """
        self._check_exportable()
        return self._compiled_model.module.export_library(*args, **kwargs)

    def export_model_library_format(self, file_name):
        """The API for exporting the bare mental build artifact in Model Library Format.

        This function creates a .tar archive containing the build artifacts in a standardized
        layout. It's intended to allow downstream automation to build TVM artifacts against the C
        runtime. It's just a simple wrapper of function tvm.micro.export_model_library_format.

        Parameters
        ----------
        file_name : str
            Path to the .tar archive to generate.
        """
        from tvm import micro  # pylint: disable=import-outside-toplevel

        self._check_exportable()
        micro.export_model_library_format(self._compiled_model, file_name=file_name)

    def create_execution_engine(self, rpc_sess=None, devices=None, with_profile=False, **kwargs):
        """The helper function to create an execution engine from the object of
        class deployable.Deployable directly.

        Parameters
        ----------
        rpc_sess : tvm.rpc.RPCSession
            The RPC session that is already connected to the RPC server.
            If it is set, the deployable object will be exported and uploaded to the RPC server
            automatically, and the remaining keyword arguments "kwargs" will be passed to the
            invocation of API "export".

        devices : tvm._ffi.runtime_ctypes.Device or list of tvm._ffi.runtime_ctypes.Device
            The devices on which to execute the compiled NN model. If it isn't set, rpc_sess.cpu(0)
            will be used if the "rpc_sess" is set, and tvm.cpu(0) will be used if the "rpc_sess"
            isn't set.

        with_profile : bool
            Whether select the execution engine that with profiling ability or not. If it is True,
            the underlying class of execution engine will use "VirtualMachineDebug" or
            "GraphExecutorDebug" instead of "VirtualMachine" or "GraphExecutor".

        Returns
        -------
        result : ExecutionEngine
            The instance of class ExecutionEngine can be used to execute the compiled NN model.

        Note
        -------
        For more RPC information, please refer to
        https://tvm.apache.org/docs/tutorial/cross_compilation_and_rpc.html
        """
        compiled_model = self._compiled_model.module
        if rpc_sess:
            compiled_model = AipuCompassConfig.get().deploy_file
            self.export(compiled_model, **kwargs)

        return ExecutionEngine(compiled_model, rpc_sess, devices, with_profile)
