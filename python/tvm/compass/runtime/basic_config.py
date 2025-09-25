# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Basic Configuration of Zhouyi Compass."""
import tvm
from . import _ffi_api


@tvm.register_object("compass.runtime.CompassBasicConfig")
class CompassBasicConfig(tvm.Object):
    """The basic Compass configuration."""

    @staticmethod
    def get():
        """The static function that is used to get the global Zhouyi NPU Compass basic configuration
        singleton.

        Returns
        -------
        result : CompassBasicConfig
            The global Zhouyi NPU Compass basic configuration singleton.
        """
        return _ffi_api.CompassBasicConfig_Global()

    @property
    def common(self):
        return _ffi_api.CompassBasicConfig_GetCommon(self)

    @property
    def runtime(self):
        return _ffi_api.CompassBasicConfig_GetRuntime(self)
