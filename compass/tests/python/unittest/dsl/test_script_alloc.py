# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


@S.prim_func
def alloc_func(inp: S.ptr("int32", "global"), out: S.ptr("int32", "global")):
    sram_1 = S.alloc([24], "int32", scope="lsram")
    for i in range(24):
        sram_1[i] = 1

    sram = S.alloc_buffer([24], "int32", scope="lsram")
    sram_ptr = sram.addr_of(0)
    S.dma_copy(sram_ptr, inp, 8)

    sram_ptr = S.alloc([4], "int32", scope="lsram")
    sram_ptr = S.alloc([8], "int32", scope="lsram")
    S.dma_copy(sram_ptr, inp + 8, 8)
    S.dma_copy(out, sram_ptr, 8)
    S.dma_copy(sram.addr_of(8), out, 8)

    sram_ptr = S.alloc_buffer([16], "int32", scope="lsram").addr_of(8)
    S.dma_copy(sram_ptr, inp + 16, 8)
    S.barrier()
    S.dma_copy(out, sram_ptr, 8)
    S.dma_copy(sram.addr_of(16), out, 8)

    for i in range(24):
        sram[i] += sram_1[i]

    S.barrier()
    S.dma_copy(out, sram, 24)


def test_alloc():
    shape = (24,)
    dtype = "int32"
    a = rand(shape, dtype)
    gt_out = a + 1

    bm = BuildManager()
    ex = bm.build(alloc_func)

    py_out = np.empty(shape, "int32")
    alloc_func(a, py_out)
    assert_allclose(gt_out, py_out)

    npu_out = np.empty(shape, "int32")
    ex(a, npu_out)
    assert_allclose(gt_out, npu_out)


if __name__ == "__main__":
    test_alloc()
