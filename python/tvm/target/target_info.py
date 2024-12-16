# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Various information about AIPU configuration."""
import tvm
from tvm import runtime
from . import _ffi_api


@tvm.register_object
class AipuInfo(runtime.Object):
    """Hardware information of AIPU target."""

    @staticmethod
    def get(target):
        """Get the predefined instance that corresponding to the given target.

        All of the instances corresponding to the supported AIPU configurations
        are created in advance, the "mcpu" attribute of the given target will be
        used to find the corresponding instance.

        Parameters
        ----------
        target : Union[str, tvm.target.Target]
            The AIPU target whose attribute "mcpu" record its AIPU configuration
            name. It can be a literal target string or a tvm.target.Target
            object.

        Returns
        -------
        aipu_info : aipu.target.AipuInfo
            The found instance.
        """
        from tvm.aipu.utils import canonicalize_target  # pylint: disable=import-outside-toplevel

        return _ffi_api.AipuInfo_Get(canonicalize_target(target))

    @classmethod
    def current(cls, allow_none=True):
        """Get the predefined instance that corresponding to the target in current context.

        Parameters
        ----------
        allow_none : bool
            Whether allow the current context haven't defined any target.

        Returns
        -------
        aipu_info : aipu.target.AipuInfo
            The found instance.
        """
        cur_target = tvm.target.Target.current(allow_none)
        if cur_target is not None and cur_target.kind.name == "aipu":
            return cls.get(cur_target)
        return None

    @property
    def is_x1(self):
        return self.name.startswith("X1")

    @property
    def is_x2(self):
        return self.name.startswith("X2")

    @property
    def is_x3(self):
        return self.name.startswith("X3")

    @property
    def is_v1(self):
        return self.is_x1

    @property
    def is_v2(self):
        return self.is_x2

    @property
    def is_v3(self):
        return self.is_x3

    @property
    def vector_width(self):
        """Hardware vector width in bits.

        Each TEC contains one vector processor unit, and the width of them are
        all same.

        Returns
        -------
        width : int
            The vector width of corresponding AIPU configuration.
        """
        return _ffi_api.AipuInfo_VectorWidth(self)

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
        return _ffi_api.AipuInfo_LsramSize(self, piece_idx)

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
        return _ffi_api.AipuInfo_GsramSize(self, piece_idx)


class Environment(object):
    """Hardware configuration object."""

    # memory scopes
    l0_scope = "lsram.0"
    l1_scope = "lsram.1"
    g0_scope = "gsram.0"
    g1_scope = "gsram.1"


# TODO(@aipu-team): different memory info according to target mcpu.
# The memory information for the compiler
@tvm.register_func("tvm.info.mem.%s" % Environment.l0_scope)
def mem_info_l0_buffer():
    return tvm.ir.make_node(
        "MemoryInfo",
        unit_bits=8,
        max_simd_bits=256,
        max_num_bits=32 * 1024 * 8 * 4,
        head_address=None,
    )


@tvm.register_func("tvm.info.mem.%s" % Environment.l1_scope)
def mem_info_l1_buffer():
    return tvm.ir.make_node(
        "MemoryInfo",
        unit_bits=8,
        max_simd_bits=256,
        max_num_bits=28 * 1024 * 8 * 4,
        head_address=None,
    )


# The memory information for the compiler
@tvm.register_func("tvm.info.mem.%s" % Environment.g0_scope)
def mem_info_g0_buffer():
    return tvm.ir.make_node(
        "MemoryInfo",
        unit_bits=8,
        max_simd_bits=256,
        max_num_bits=256 * 1024 * 8,
        head_address=None,
    )


@tvm.register_func("tvm.info.mem.%s" % Environment.g1_scope)
def mem_info_g1_buffer():
    return tvm.ir.make_node(
        "MemoryInfo",
        unit_bits=8,
        max_simd_bits=256,
        max_num_bits=256 * 1024 * 8,
        head_address=None,
    )
