# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import rand


def gen_ddr2lsram_stride(dtype):
    @S.prim_func
    def func_ddr2lsram_stride(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global")):
        lsram = S.alloc_buffer(32, dtype, "lsram")
        S.dma_transpose2d(lsram, a, 4, 8, 4, 10)
        S.dma_copy(b, lsram, 32)

    return func_ddr2lsram_stride


def gen_ddr2ddr(dtype, sram_type):
    @S.prim_func
    def func_ddr2ddr(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), col: S.int32, row: S.int32):
        S.dma_transpose2d(b, a, row, col)

    return func_ddr2ddr


def gen_sram2ddr(dtype, sram_type):
    @S.prim_func
    def func_sram2ddr(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), col: S.int32, row: S.int32):
        sram = S.alloc_buffer(8 * 1024, dtype, sram_type)
        n = col * row
        # copy: a -> sram
        # tranpose: sram ->b
        S.dma_copy(sram, a, n)
        S.dma_transpose2d(b, sram, row, col)

    return func_sram2ddr


def gen_ddr2sram(dtype, sram_type):
    @S.prim_func
    def func_ddr2sram(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), col: S.int32, row: S.int32):
        sram = S.alloc_buffer(8 * 1024, dtype, sram_type)
        n = col * row
        # tranpose: a ->sram
        # copy: sram -> b
        S.dma_transpose2d(sram, a, row, col)
        S.dma_copy(b, sram, n)

    return func_ddr2sram


def verify_dma_transpose2d(gen_func, dtype, col, row, sram_type=None):
    func = gen_func(dtype, sram_type)

    size = row * col
    a = rand(size, dtype)
    gt_out = np.transpose(a.reshape(row, col)).flatten()

    bm = aipu.tir.BuildManager()
    ex = bm.build(func)

    py_out = np.empty(size, dtype=dtype)
    func(a, py_out, col, row)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(size, dtype=dtype)
    ex(a, aipu_out, col, row)
    testing.assert_allclose(aipu_out, gt_out)

    # print(ex.c_code)
    # print(a.reshape(row, col))
    # print(aipu_out.reshape(col, row))


def test_direction():
    verify_dma_transpose2d(gen_sram2ddr, "int8", 8, 8, "lsram")
    verify_dma_transpose2d(gen_ddr2sram, "int8", 8, 8, "lsram")
    verify_dma_transpose2d(gen_sram2ddr, "int8", 8, 8, "shared")
    verify_dma_transpose2d(gen_ddr2sram, "int8", 8, 8, "shared")
    verify_dma_transpose2d(gen_ddr2ddr, "int8", 8, 8)


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_dtype(dtype):
    verify_dma_transpose2d(gen_sram2ddr, dtype, 128, 8, "lsram")


@pytest.mark.parametrize("size", (8, 1024, 0xFFFF))
def test_size(size):
    verify_dma_transpose2d(gen_ddr2ddr, "int8", size, 16)
    verify_dma_transpose2d(gen_ddr2ddr, "int16", size, 16)
    verify_dma_transpose2d(gen_ddr2ddr, "float32", size, 16)

    # test size* item_byte  <0xffff
    # verify_dma_transpose2d(gen_ddr2ddr, "int8", 0xFFFF, 2)
    # verify_dma_transpose2d(gen_ddr2ddr, "int16", 0x7FFF, 2)
    # verify_dma_transpose2d(gen_ddr2ddr, "float32", 0x3FFF, 2)

    # test size *item_size <0xffffff (cut on row)
    # verify_dma_transpose2d(gen_ddr2ddr, "int8", 2, 0xFFFFFF)
    # verify_dma_transpose2d(gen_ddr2ddr, "int16", 2, 0xFFFFF)
    # verify_dma_transpose2d(gen_ddr2ddr, "float32", 2, 0xFFFFF)

    # test size *item_size <0xffffff (cut on col)
    # verify_dma_transpose2d(gen_ddr2ddr, "int8", 0xFFFFFF, 2)
    # verify_dma_transpose2d(gen_ddr2ddr, "int16", 0xFFFFF, 2)
    # verify_dma_transpose2d(gen_ddr2ddr, "float32", 0xFFFFF, 2)


def test_transpose_stride():
    dtype = "int8"
    func = gen_ddr2lsram_stride("int8")

    a = np.array(range(40), dtype)
    gt_out = np.transpose(a.reshape(4, 10)[:, :8]).flatten()

    bm = aipu.tir.BuildManager()
    ex = bm.build(func)

    py_out = np.empty(32, dtype=dtype)
    func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(32, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)

    # print(ex.c_code)
    # print(a.reshape(4, 10))
    # print(aipu_out.reshape(8, 4))


if __name__ == "__main__":
    verify_dma_transpose2d(gen_sram2ddr, "int8", 8, 8, "lsram")
    verify_dma_transpose2d(gen_ddr2sram, "int8", 8, 8, "lsram")
    verify_dma_transpose2d(gen_sram2ddr, "int8", 8, 8, "shared")
    verify_dma_transpose2d(gen_ddr2sram, "int8", 8, 8, "shared")
    verify_dma_transpose2d(gen_ddr2ddr, "int8", 8, 8)
    test_transpose_stride()
