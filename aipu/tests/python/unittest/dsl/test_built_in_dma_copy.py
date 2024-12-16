# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import rand


_DMA_MAX_WIDTH = 0xFFFF  # 65535
_DMA_MAX_STRIDE = 0xFFFFFF  # 16777215
_DMA_MAX_TRANS_SIZE = 0xFFFFFF  # 16777215
dtype = "int8"


def gen_func_case0(width, stride, times, out_n):
    @S.prim_func
    def func_dma_copy(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        tid = S.get_local_id()
        if tid == 0:
            a_lsram = S.alloc_buffer(dtype=dtype, shape=(out_n), scope="lsram")
            S.dma_copy(a_lsram, a, width, src_stride=stride, times=times)
            S.dma_copy(b, a_lsram, width, times=times, dst_stride=stride)
            S.dma_copy(c, b, width, src_stride=stride, times=times)

    return func_dma_copy


# EXT2INT(ddr->lsram):
#   src_stride > _DMA_MAX_STRIDE
# INT2EXT(lsram->ddr):
#   dst_stride > _DMA_MAX_STRIDE
# INT2EXT(ddr->ddr):
#   src_stride > _DMA_MAX_STRIDE
#   32 <= width < _DMA_MAX_WIDTH
def test_unit_case0():
    width = 32
    times = 4
    stride = _DMA_MAX_STRIDE + 10
    n = stride * times
    out_n = width * times
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = []
    for i in range(times):
        gt_out.append(a[i * stride : i * stride + width])
    gt_out = np.concatenate(gt_out)

    bm = aipu.tir.BuildManager()
    prim_func = gen_func_case0(width, stride, times, out_n)
    ex = bm.build(prim_func)

    py_out = np.empty(out_n, dtype=dtype)
    prim_func(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(out_n, dtype=dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_func_case1(width, times):
    @S.prim_func
    def func_dma_copy(a: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        S.dma_copy(c, a, width, times=times)

    return func_dma_copy


# INT2EXT:
#   width > _DMA_MAX_WIDTH
#   width == src_stride == dst_stride
#   size < _DMA_MAX_TRANS_SIZE
def test_unit_case1():
    width = _DMA_MAX_WIDTH + 100
    times = _DMA_MAX_TRANS_SIZE // width
    n = width * times
    a = rand(n, dtype)
    gt_out = a

    bm = aipu.tir.BuildManager()
    prim_func = gen_func_case1(width, times)
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype=dtype)
    prim_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_func_case2(width, times):
    @S.prim_func
    def func_dma_copy(a: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        S.dma_copy(c, a, width, times=times)

    return func_dma_copy


# INT2EXT:
#   width > _DMA_MAX_WIDTH
#   width == src_stride == dst_stride
#   size > _DMA_MAX_TRANS_SIZE
def test_unit_case2():
    width = _DMA_MAX_WIDTH + 100
    times = _DMA_MAX_TRANS_SIZE // width + 1
    n = width * times
    a = rand(n, dtype)
    gt_out = a

    bm = aipu.tir.BuildManager()
    prim_func = gen_func_case2(width, times)
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype=dtype)
    prim_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_func_case3(width, stride, times):
    @S.prim_func
    def func_dma_copy(a: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        S.dma_copy(c, a, width, src_stride=stride, times=times)

    return func_dma_copy


# INT2EXT:
#   src_stride > _DMA_MAX_STRIDE
#   width < 32
def test_unit_case3():
    width = 16
    stride = _DMA_MAX_STRIDE + 10
    times = 4
    n = stride * times
    out_n = width * times
    a = rand(n, dtype)
    gt_out = []
    for i in range(times):
        gt_out.append(a[i * stride : i * stride + width])
    gt_out = np.concatenate(gt_out)

    bm = aipu.tir.BuildManager()
    prim_func = gen_func_case3(width, stride, times)
    ex = bm.build(prim_func)

    py_out = np.empty(out_n, dtype=dtype)
    prim_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(out_n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_func_case4(width, stride, times):
    @S.prim_func
    def func_dma_copy(a: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        S.dma_copy(c, a, width, src_stride=stride, times=times)

    return func_dma_copy


# INT2EXT:
#   _DMA_MAX_WIDTH < width < _DMA_MAX_TRANS_SIZE
#   src_stride <= _DMA_MAX_STRIDE
#   dst_stride <= _DMA_MAX_STRIDE
#   width != src_stride
def test_unit_case4():
    width = _DMA_MAX_WIDTH + 100
    src_stride = width + 100
    times = 4
    n = src_stride * times
    out_n = width * times
    a = rand(n, dtype)
    gt_out = []
    for i in range(times):
        gt_out.append(a[i * src_stride : i * src_stride + width])
    gt_out = np.concatenate(gt_out)

    bm = aipu.tir.BuildManager()
    prim_func = gen_func_case4(width, src_stride, times)
    ex = bm.build(prim_func)

    py_out = np.empty(out_n, dtype=dtype)
    prim_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(out_n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_func_case5(width, stride, times):
    @S.prim_func
    def func_dma_copy(a: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        S.dma_copy(c, a, width, src_stride=stride, times=times)

    return func_dma_copy


# INT2EXT:
#   width > _DMA_MAX_TRANS_SIZE > _DMA_MAX_WIDTH
#   src_stride > _DMA_MAX_STRIDE
#   dst_stride > _DMA_MAX_STRIDE
#   width != src_stride
def test_unit_case5():
    width = _DMA_MAX_TRANS_SIZE + 100
    src_stride = width + 100
    times = 4
    n = src_stride * times
    out_n = width * times
    a = rand(n, dtype)
    gt_out = []
    for i in range(times):
        gt_out.append(a[i * src_stride : i * src_stride + width])
    gt_out = np.concatenate(gt_out)

    bm = aipu.tir.BuildManager()
    prim_func = gen_func_case5(width, src_stride, times)
    ex = bm.build(prim_func)

    py_out = np.empty(out_n, dtype=dtype)
    prim_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(out_n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_func_case6(width, stride, times):
    @S.prim_func
    def func_dma_copy(a: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        S.dma_copy(c, a, width, src_stride=stride, times=times)

    return func_dma_copy


# INT2EXT:
#   size > _DMA_MAX_TRANS_SIZE
#   width <= _DMA_MAX_WIDTH
#   src_stride <= _DMA_MAX_STRIDE
#   dst_stride <= _DMA_MAX_STRIDE
#   width != src_stride
def test_unit_case6():
    width = _DMA_MAX_WIDTH
    src_stride = width + 100
    times = _DMA_MAX_TRANS_SIZE // width + 1
    n = src_stride * times
    out_n = width * times
    a = rand(n, dtype)
    gt_out = []
    for i in range(times):
        gt_out.append(a[i * src_stride : i * src_stride + width])
    gt_out = np.concatenate(gt_out)

    bm = aipu.tir.BuildManager()
    prim_func = gen_func_case6(width, src_stride, times)
    ex = bm.build(prim_func)

    py_out = np.empty(out_n, dtype=dtype)
    prim_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(out_n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_func_case7(width):
    @S.prim_func
    def func_dma_copy(a: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        a_shared = S.alloc_buffer(dtype=dtype, shape=(width), scope="shared")
        S.dma_copy(a_shared, a, width)
        S.dma_copy(c, a_shared, width)

    return func_dma_copy


# EXT2INT(ddr->shared):
#   width > _DMA_MAX_WIDTH
def test_unit_case7():
    n = _DMA_MAX_WIDTH + 100
    a = rand(n, dtype)
    gt_out = a

    bm = aipu.tir.BuildManager()
    prim_func = gen_func_case7(n)
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype=dtype)
    prim_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_func_case8(width, times, out_n):
    @S.prim_func
    def func_dma_copy(a: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        S.dma_memset(c, 1, out_n)
        S.dma_copy(c, a, width, times=times)

    return func_dma_copy


# INT2EXT:
#   width < _DMA_MAX_WIDTH
#   width == src_stride == dst_stride
#   size > _DMA_MAX_TRANS_SIZE
#   (size - _DMA_MAX_TRANS_SIZE) < _DMA_MAX_WIDTH
def test_unit_case8():
    width = 28
    times = _DMA_MAX_TRANS_SIZE // width + 1
    n = times * width
    a = rand(n, dtype)
    out_n = _DMA_MAX_TRANS_SIZE + 29
    gt_out = np.concatenate([a, np.ones(out_n - n, dtype=dtype)])

    bm = aipu.tir.BuildManager()
    prim_func = gen_func_case8(width, times, out_n)
    ex = bm.build(prim_func)

    py_out = np.empty(out_n, dtype=dtype)
    prim_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(out_n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_func_case9(width, src_stride, dst_stride, times, out_n):
    @S.prim_func
    def func_dma_copy(a: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        a_shared = S.alloc_buffer(dtype=dtype, shape=(out_n), scope="shared")
        S.dma_memset(a_shared, 1, out_n)
        S.dma_copy(a_shared, a, width, src_stride=src_stride, times=times, dst_stride=dst_stride)
        S.dma_copy(c, a_shared, out_n)

    return func_dma_copy


# EXT2INT(ddr->shared):
#   width < _DMA_MAX_WIDTH
#   src_stride > _DMA_MAX_STRIDE
def test_unit_case9():
    width = _DMA_MAX_WIDTH - 100
    times = 2
    src_stride = _DMA_MAX_STRIDE + 100
    n = src_stride * times
    dst_stride = width + 5
    out_n = dst_stride * times
    a = rand(n, dtype)
    gt_out = np.ones(out_n, dtype=dtype)
    for i in range(times):
        gt_out[i * dst_stride : i * dst_stride + width] = a[i * src_stride : i * src_stride + width]

    bm = aipu.tir.BuildManager()
    prim_func = gen_func_case9(width, src_stride, dst_stride, times, out_n)
    ex = bm.build(prim_func)

    py_out = np.empty(out_n, dtype=dtype)
    prim_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(out_n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_unit_case0()
    test_unit_case1()
    test_unit_case2()
    test_unit_case3()
    test_unit_case4()
    test_unit_case5()
    test_unit_case6()
    test_unit_case7()
    test_unit_case8()
    test_unit_case9()
