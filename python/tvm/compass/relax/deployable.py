# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Code to handle the work when deploying a TVM compiled NN model."""
import os
import glob
import shutil
from .config import CompassConfig
from .execution_engine import ExecutionEngine


class Deployable:
    """The Relax build result for VM.

    This class should only be instanced by the compilation Python API.
    """

    def __init__(self, compiled_model):
        self._compiled_model = compiled_model

    def export(self, dir_path, fcompile=None, **kwargs):
        """Exporting the compiled NN model as several files for deployment.

        Parameters
        ----------
        dir_path : str
            The destination directory in where the exported files will be stored.

        fcompile : function(target, file_list, kwargs), optional
            The compilation function to use create the final library object during export.
            For example, when fcompile=_cc.create_shared, or when it is not supplied but module is
            "llvm," this is used to link all produced artifacts into a final dynamic library.
            If fcompile has attribute object_format, will compile host library to that format.
            Otherwise, will use default format "o".
        """
        msg = f'The path "{dir_path}" is not a directory.'
        assert not os.path.exists(dir_path) or os.path.isdir(dir_path), msg

        if self._compiled_model._collect_from_import_tree(
            lambda x: x.type_key == "compass.runtime.PackedFuncModule"
        ):
            msg = "Can't export the deployable instance, because Compass engine isn't set to "
            raise RuntimeError(msg + '"driver" when compiling the NN model.')

        output_dir = CompassConfig.get().common["output_dir"]
        for pattern in ("**/aipu.bin", "**/temp.graph.json", "**/aipu.bin.extraweight*"):
            for src_file in glob.iglob(f"{output_dir}/{pattern}", recursive=True):
                dst_file = f"{dir_path}/{os.path.relpath(src_file, start=output_dir)}"
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy(src_file, dst_file)

        library_obj_path = f"{dir_path}/nn_model."
        library_obj_path += fcompile.output_format if hasattr(fcompile, "output_format") else "so"
        self._compiled_model.export_library(library_obj_path, fcompile=fcompile, **kwargs)

    def create_execution_engine(self, rpc_sess=None, devices=None, with_profile=False, **kwargs):
        """The helper function to create an execution engine directly.

        Parameters
        ----------
        rpc_sess : tvm.rpc.RPCSession
            The RPC session that is already connected to the RPC server.
            If it is set, the deployable object will be exported and uploaded to the RPC server
            automatically, and the remaining keyword arguments "kwargs" will be passed to the
            invocation of API "export".

        devices : tvm.runtime.Device or list of tvm.runtime.Device
            The devices on which to execute the compiled NN model. If it isn't set, rpc_sess.cpu(0)
            will be used if the "rpc_sess" is set, and tvm.cpu(0) will be used if the "rpc_sess"
            isn't set.

        with_profile : bool
            Whether select the execution engine that with profiling ability or not. If it is True,
            the attribute "executor" of the result execution engine will enable profiling.

        Returns
        -------
        result : ExecutionEngine
            The instance of class ExecutionEngine can be used to execute the compiled NN model.

        Note
        -------
        For more RPC information, please refer to
        https://tvm.apache.org/docs/tutorial/cross_compilation_and_rpc.html
        """
        compiled_model = self._compiled_model
        if rpc_sess:
            compiled_model = CompassConfig.get().deploy_dir
            self.export(compiled_model, **kwargs)
        return ExecutionEngine(compiled_model, rpc_sess, devices, with_profile)
