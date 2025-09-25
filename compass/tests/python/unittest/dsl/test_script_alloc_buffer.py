# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


@S.prim_func
def dma_copy_func(src: S.ptr("int32", "global"), dst: S.ptr("int32", "global"), n: S.int32):
    for i in S.tec_range(4):
        sram = S.alloc_buffer([1024], "int32", scope="lsram")
        tec_move_items = (n + 3) >> 2
        move_items = S.min(n - i * tec_move_items, tec_move_items)
        loop_num = (move_items + 1023) // 1024
        for loop in range(loop_num):
            loop_item = S.min(move_items - loop * 1024, 1024)
            base = i * tec_move_items + 1024 * loop
            S.dma_copy(sram, src + base, loop_item)
            S.dma_copy(dst + base, sram, loop_item)


def test_alloc_buffer():
    inp_shape = tuple(rand(4, "int32", 10, 20))
    inp_array = rand(inp_shape, "int32", -1000, 1000)
    n = inp_array.size

    bm = BuildManager()
    ex = bm.build(dma_copy_func)

    py_out = np.empty(inp_shape, "int32")
    dma_copy_func(inp_array, py_out, n)
    assert_allclose(inp_array, py_out)

    npu_out = np.empty(inp_shape, "int32")
    ex(inp_array, npu_out, n)
    assert_allclose(inp_array, npu_out)


if __name__ == "__main__":
    test_alloc_buffer()
