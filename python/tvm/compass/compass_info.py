# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Various information about Zhouyi NPU configuration."""
import tvm
from . import _ffi_api
from .utils import canonicalize_target


@tvm.register_object("compass.CompassInfo")
class CompassInfo(tvm.Object):
    """Hardware information of Zhouyi NPU target."""

    @staticmethod
    def get(target):
        """Get the predefined instance that corresponding to the given target.

        All of the instances corresponding to the supported Zhouyi NPU configurations are created in
        advance, the "mcpu" attribute of the given target will be used to find the corresponding
        instance.

        Parameters
        ----------
        target : Union[str, tvm.target.Target]
            The Zhouyi NPU target whose attribute "mcpu" record its configuration name. It can be a
            literal target string or a ``tvm.target.Target`` object.

        Returns
        -------
        cps_info : compass.CompassInfo
            The found instance.
        """

        return _ffi_api.CompassInfo_Get(canonicalize_target(target))

    @classmethod
    def current(cls, allow_none=True):
        """Get the predefined instance that corresponding to the target in current context.

        Parameters
        ----------
        allow_none : bool
            Whether allow the current context haven't defined any target.

        Returns
        -------
        cps_info : compass.CompassInfo
            The found instance.
        """
        cur_target = tvm.target.Target.current(allow_none)
        if cur_target is not None and cur_target.kind.name == "compass":
            return cls.get(cur_target)
        return None

    @property
    def vector_width(self):
        """Hardware vector width in bits.

        Each TEC contains one vector processor unit, and the width of them are
        all same.

        Returns
        -------
        width : int
            The vector width of corresponding Zhouyi NPU configuration.
        """
        return _ffi_api.CompassInfo_VectorWidth(self)

    def lsram_size(self, piece_idx=0):
        """The size of the given piece local SRAM in bytes for each TEC.

        Each TEC contains one or more pieces of local SRAM, size and piece count
        of the local SRAM of each TEC are same.

        Parameters
        ----------
        piece_idx : int, optional
            The local SRAM piece index.

        Returns
        -------
        size : int
            The size of the specified piece of local SRAM.
        """
        return _ffi_api.CompassInfo_LsramSize(self, piece_idx)

    def gsram_size(self, piece_idx=0):
        """The size of the given piece global SRAM in bytes for each core.

        Each core contains one or more pieces of global SRAM, they are shared by
        all of the TECs in the same core.

        Parameters
        ----------
        piece_idx : int, optional
            The global SRAM piece index.

        Returns
        -------
        size : int
            The size of the specified piece of global SRAM.
        """
        return _ffi_api.CompassInfo_GsramSize(self, piece_idx)
