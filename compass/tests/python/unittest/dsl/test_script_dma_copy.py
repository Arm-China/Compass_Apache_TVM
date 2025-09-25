# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


dtype = "int16"


@S.prim_func
def func_dma_copy(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
    aa = S.alloc_buffer(dtype=dtype, shape=[16], scope="lsram")
    bb = S.alloc_buffer(dtype=dtype, shape=[16], scope="lsram")
    cc = S.alloc_buffer(dtype=dtype, shape=[16], scope="lsram")

    S.dma_copy(aa, a, 16)
    S.dma_copy(bb, b, 16)

    for i in range(16):
        cc[i] = aa[i] + bb[i]

    S.dma_copy(c, cc, 16)


def test_dma_copy():
    a = rand(16, dtype)
    b = rand(16, dtype)
    gt_out = a + b

    bm = BuildManager()
    ex = bm.build(func_dma_copy)

    py_out = np.empty(16, dtype=dtype)
    func_dma_copy(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(16, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


@S.prim_func
def func_pointer_dma_copy(A: S.ptr(dtype, "global"), B: S.ptr(dtype, "global"), C: S.ptr(dtype, "global")):
    FULL_SIZE = S.meta_var(48)
    lsram = S.alloc_buffer(dtype=dtype, shape=[FULL_SIZE], scope="lsram")

    S.dma_copy(lsram, A, 16)
    S.dma_copy(lsram.addr_of(16), B, 16)

    for i in range(16):
        lsram[i + 32] = lsram[i] + lsram[i + 16]
    S.dma_copy(C, lsram.addr_of(32), 16)


def test_pointer_dma_copy():
    a = rand(16, dtype)
    b = rand(16, dtype)
    gt_out = a + b

    bm = BuildManager()
    ex = bm.build(func_pointer_dma_copy)

    py_out = np.empty(16, dtype=dtype)
    func_pointer_dma_copy(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(16, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


@S.prim_func
def func_ddr2ddr_dma_copy(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
    ev0 = S.alloc_events(1)
    S.async_dma_copy(c, a, 8, event=ev0)
    S.dma_copy(c + 8, b, 8)

    S.wait_events(ev0)
    S.free_events(ev0)


def test_ddr2ddr_dma_copy():
    a = rand(16, dtype)
    b = rand(16, dtype)
    gt_out = np.concatenate((a[:8], b[:8]))

    bm = BuildManager()
    ex = bm.build(func_ddr2ddr_dma_copy)

    py_out = np.empty(16, dtype=dtype)
    func_ddr2ddr_dma_copy(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(16, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_dma_copy()
    test_pointer_dma_copy()
    test_ddr2ddr_dma_copy()
