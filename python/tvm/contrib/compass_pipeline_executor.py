# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Compass pipelines modules."""
from tvm._ffi import get_global_func
from .pipeline_executor import PipelineModule


class CompassPipelineModule(PipelineModule):
    """
    Wrapper for PipelineModule class
    override load_library function to create shared_config file
    """

    @staticmethod
    def load_library(config_file_name):
        """Import files to create a pipeline executor.

        Parameters
        ----------
        config_file_name : str
            Path and name of the configuration file, the configuration file contains the
            disk path of the parameter file, library file, and JSON file.
        """

        # Load a PipelineExecutor from the disk files.
        load_library = get_global_func("compass_pipeline.load", False)
        module = load_library(config_file_name)

        return CompassPipelineModule(module)
