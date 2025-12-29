# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


# size = 1000639
size = 5 * 1024 + 1


def gen_func_dma_memset_ddr_value_num_imm(dtype, value):
    @S.prim_func
    def func_dma_memset_imm(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global")):
        S.dma_memset(a, value, size)
        for i in range(size):
            b[i] = a[i]

    return func_dma_memset_imm


def gen_func_dma_memset_ddr_value_num_var(dtype):
    @S.prim_func
    def func_dma_memset_var(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), batch: S.int32, innersize: S.int32):
        value = a[0]
        S.dma_memset(a, value, batch * innersize)
        for i in range(batch * innersize):
            b[i] = a[i]

    return func_dma_memset_var


def gen_func_dma_memset_sram(dtype, value, size, dtype_lanes=1):
    @S.prim_func
    def func_dma_memset_sram(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global")):
        lsram = S.alloc_buffer([size], dtype, scope="lsram")
        S.dma_memset(lsram, value, size * dtype_lanes)
        for i in range(size):
            b[i] = a[i] + lsram[i]

    return func_dma_memset_sram


def verify(prim_func, a, gt_out, dtype, size):
    bm = BuildManager()
    ex = bm.build(prim_func)
    # print(ex.c_code)
    py_out = np.empty(size, dtype=dtype)
    prim_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(size, dtype=dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)
    # print(npu_out)


def verify_num_var(prim_func, a, gt_out, dtype, batch, innersize):
    size = batch * innersize

    bm = BuildManager()
    ex = bm.build(prim_func)
    # print(ex.c_code)

    py_out = np.empty(size, dtype=dtype)
    prim_func(a, py_out, batch, innersize)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(size, dtype=dtype)
    ex(a, npu_out, batch, innersize)
    assert_allclose(npu_out, gt_out)
    # print(npu_out[:5])


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_dma_memset_ddr(dtype):
    value = 10.0
    if dtype.startswith("int") or dtype.startswith("uint"):
        value = int(value)

    a = rand((size,), dtype)
    a[0] = np.array([value], dtype=dtype)[0]

    gt_out = np.ones((size,), dtype) * value

    # value,num as imm
    prim_func0 = gen_func_dma_memset_ddr_value_num_imm(dtype, value)
    verify(prim_func0, a, gt_out, dtype, size)

    # value, num as var
    batch = 2
    innersize = 1024
    gt_out1 = np.ones((batch * innersize,), dtype) * value
    prim_func1 = gen_func_dma_memset_ddr_value_num_var(dtype)
    verify_num_var(prim_func1, a, gt_out1, dtype, batch, innersize)


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_dma_memset_sram(dtype):
    size = 5 * 1024 + 1
    value = 10.0
    if dtype.startswith("int") or dtype.startswith("uint"):
        value = int(value)

    a = rand((size,), dtype)
    gt_out = a + np.ones((size,), dtype) * value

    # scalar
    prim_func = gen_func_dma_memset_sram(dtype, value, size)
    verify(prim_func, a, gt_out, dtype, size)

    # vector
    vdtype = hw_native_vdtype(dtype)
    size = 10
    a = rand((size * vdtype.lanes,), dtype)
    gt_out = a + np.ones((size * vdtype.lanes,), dtype) * value

    prim_func = gen_func_dma_memset_sram(vdtype, value, size, vdtype.lanes)
    verify(prim_func, a, gt_out, dtype, size * vdtype.lanes)


if __name__ == "__main__":
    test_dma_memset_ddr("uint8")
    test_dma_memset_sram("float32")
