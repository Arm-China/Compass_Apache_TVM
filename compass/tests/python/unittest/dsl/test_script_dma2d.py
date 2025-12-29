# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose

dtype = "int32"
shape_2d = [11, 11]
shape_2d_size = 11 * 11
slice_shape = [10, 10]


@S.prim_func
def func_dma_copy(A: S.ptr(dtype, "global"), B: S.ptr(dtype, "global"), C: S.ptr(dtype, "global")):
    a = S.match_buffer(A, shape=shape_2d)
    b = S.match_buffer(B, shape=shape_2d)
    c = S.match_buffer(C, shape=shape_2d)

    aa = S.alloc_buffer(dtype=dtype, shape=shape_2d, scope="lsram")
    bb = S.alloc_buffer(dtype=dtype, shape=shape_2d, scope="lsram")
    cc = S.alloc_buffer(dtype=dtype, shape=shape_2d, scope="lsram")

    S.dma_copy(aa, a, shape_2d_size)
    S.dma_copy(bb, b, shape_2d_size)

    for i in range(shape_2d[1]):
        for j in range(shape_2d[0]):
            cc[j, i] = aa[j, i] + bb[j, i]

    S.dma_copy(c, cc, shape_2d_size)


@S.prim_func
def func_stride_dma_copy(A: S.ptr(dtype, "global"), B: S.ptr(dtype, "global"), C: S.ptr(dtype, "global")):
    a = S.match_buffer(A, shape=shape_2d)
    b = S.match_buffer(B, shape=shape_2d)
    c = S.match_buffer(C, shape=shape_2d)

    aa = S.alloc_buffer(dtype=dtype, shape=slice_shape, scope="lsram")
    bb = S.alloc_buffer(dtype=dtype, shape=slice_shape, scope="lsram")
    cc = S.alloc_buffer(dtype=dtype, shape=slice_shape, scope="lsram")

    S.dma_copy(aa, a.addr_of([1, 1]), 10, 11, 10)
    S.dma_copy(bb, b, 10, 11, 10)

    for i in range(slice_shape[1]):
        for j in range(slice_shape[0]):
            cc[j, i] = aa[j, i] + bb[j, i]
    S.dma_copy(c.addr_of([1, 1]), cc, 10, 10, 10, 11)


def test_dma_copy():
    a = rand(shape_2d, dtype)
    b = rand(shape_2d, dtype)

    gt_out = a + b

    bm = BuildManager()
    ex = bm.build(func_dma_copy)

    py_out = np.empty(shape_2d, dtype=dtype)
    func_dma_copy(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(shape_2d, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


def test_stride_dma_copy():
    a = rand(shape_2d, dtype)
    b = rand(shape_2d, dtype)
    gt_out = np.zeros(shape_2d, dtype=dtype)
    gt_out[1:, 1:] = a[1:, 1:] + b[:10, :10]

    bm = BuildManager()
    ex = bm.build(func_stride_dma_copy)

    py_out = np.empty(shape_2d, dtype=dtype)
    func_stride_dma_copy(a, b, py_out)
    assert_allclose(py_out[1:, 1:], gt_out[1:, 1:])

    npu_out = np.empty(shape_2d, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out[1:, 1:], gt_out[1:, 1:])


if __name__ == "__main__":
    test_dma_copy()
    test_stride_dma_copy()
