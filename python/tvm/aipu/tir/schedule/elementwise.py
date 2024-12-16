# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Zhouyi Compass schedule for elementwise."""
from tvm import tir, DataType, target as tgt
from tvm.aipu.utils import hw_native_vdtype
from .schedule import Schedule
from ..transform import MergeForWhere


def elementwise_unary(func, target, block):
    """Schedule for unary computation."""
    io_params = [param for param in func._param_anns if isinstance(param, tir.Pointer)]
    msg = "The schedule only support a input and a output with same dtype."
    assert len(io_params) == 2 and io_params[0].dtype == io_params[1].dtype, msg
    dtype = DataType(io_params[0].dtype)
    aipu_info = tgt.AipuInfo.get(target)
    tec_num = aipu_info.tec_count
    lsram_size = aipu_info.lsram_size() // dtype.bytes
    lanes = hw_native_vdtype(dtype).lanes

    sch = Schedule(func)

    (i,) = sch.get_loops(block)

    bind, loop, sram = sch.split(i, factors=[tec_num, None, lsram_size])
    sch.bind_tec(bind)
    lsram_inp = sch.cache_read(block, 0, "lsram")
    sch.compute_at(lsram_inp, loop)
    lsram_out = sch.cache_write(block, 0, "lsram")
    sch.reverse_compute_at(lsram_out, loop)

    mod = MergeForWhere(block)(sch.mod)  # pylint: disable=not-callable
    sch = Schedule(mod)
    bind, loop, sram = sch.get_loops(block)
    _, v_index = sch.split(sram, factors=[None, lanes], disable_predication=True)
    sch.vectorize(v_index)

    return sch


def elementwise_binary(func, target, block):
    """Schedule for binary computation."""
    io_params = [param for param in func._param_anns if isinstance(param, tir.Pointer)]
    msg = "The schedule only support two input and a output with same dtype."
    dtype = io_params[0].dtype
    assert len(io_params) == 3 and all(dtype == param.dtype for param in io_params), msg
    dtype = DataType(dtype)
    aipu_info = tgt.AipuInfo.get(target)
    tec_num = aipu_info.tec_count
    lsram_size = aipu_info.lsram_size() // dtype.bytes // 2
    lanes = hw_native_vdtype(dtype).lanes

    sch = Schedule(func)

    (i,) = sch.get_loops(block)

    bind, loop, sram = sch.split(i, factors=[tec_num, None, lsram_size])
    sch.bind_tec(bind)
    lsram_inp0 = sch.cache_read(block, 0, "lsram")
    sch.compute_at(lsram_inp0, loop)
    lsram_inp1 = sch.cache_read(block, 1, "lsram")
    sch.compute_at(lsram_inp1, loop)
    lsram_out = sch.cache_write(block, 0, "lsram")
    sch.reverse_compute_at(lsram_out, loop)

    mod = MergeForWhere(block)(sch.mod)  # pylint: disable=not-callable
    sch = Schedule(mod)
    bind, loop, sram = sch.get_loops(block)
    _, v_index = sch.split(sram, factors=[None, lanes], disable_predication=True)
    sch.vectorize(v_index)

    return sch
