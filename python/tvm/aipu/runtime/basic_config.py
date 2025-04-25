# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Basic Configuration of Zhouyi Compass."""
import tvm
from . import _ffi_api


@tvm.register_object("aipu.runtime.AipuCompassBasicConfig")
class AipuCompassBasicConfig(tvm.Object):
    """The basic AIPU Compass configuration."""

    @staticmethod
    def get():
        """The static function that is used to get the global
        Zhouyi NPU Compass basic configuration singleton.

        Returns
        -------
        result : AipuCompassBasicConfig
            The global Zhouyi NPU Compass basic configuration singleton.
        """
        return _ffi_api.AipuCompassBasicConfig_Global()

    @property
    def common(self):
        return _ffi_api.AipuCompassBasicConfig_GetCommon(self)

    @property
    def runtime(self):
        return _ffi_api.AipuCompassBasicConfig_GetRuntime(self)
