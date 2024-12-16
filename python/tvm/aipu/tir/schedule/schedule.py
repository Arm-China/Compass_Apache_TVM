# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Implement schedule relevant APIs of Zhouyi Compass extension of TIR."""
from tvm import ir, tir
from tvm.tir.schedule import LoopRV, BlockRV
from ...script.parser import PyPrimFunc, parse_to_prim_func
from ...tir.analysis import extract_prim_func_info


class Schedule(tir.Schedule):
    """AIPU schedule class"""

    def __init__(self, mod, *args, **kwargs):
        if isinstance(mod, PyPrimFunc):
            py_prim_func = mod
            prim_func = parse_to_prim_func(py_prim_func)
            prim_func = prim_func.with_attr("tir.is_entry_func", True)
            mod = ir.IRModule({"main": prim_func})
            py_prim_func.param_infos = extract_prim_func_info(mod).param_infos
        super().__init__(mod, *args, **kwargs)

    def read_at(
        self, loop: LoopRV, block: BlockRV, read_buffer_index: int, storage_scope: str
    ) -> BlockRV:
        """cache_read + compute_at"""
        read_buffer = super().cache_read(block, read_buffer_index, storage_scope)
        super().compute_at(read_buffer, loop)
        return read_buffer

    def write_at(
        self, loop: LoopRV, block: BlockRV, write_buffer_index: int, storage_scope: str
    ) -> BlockRV:
        """cache_write + reverse_compute_at"""
        write_buffer = super().cache_write(block, write_buffer_index, storage_scope)
        super().reverse_compute_at(write_buffer, loop)
        return write_buffer

    def bind_tec(self, loop: LoopRV) -> None:
        """Bind the input loop to the given thread Index X."""
        super().bind(loop, thread_axis="threadIdx.x")
