# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import DataType
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vload_vstore(dtype):
    @S.prim_func
    def func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global")):
        va = S.vload(a)
        S.vstore(va, b)

    return func


def gen_vload_stride(is_stride_var, dtype, stride_imm):
    @S.prim_func
    def vload_stride_with_stride_var(
        a: S.ptr(dtype, "global"),
        stride_var: S.i32,
        b: S.ptr(dtype, "global"),
    ):
        va = S.vload(a, stride=stride_var)
        S.vstore(va, b)

    @S.prim_func
    def vload_stride_with_stride_imm(
        a: S.ptr(dtype, "global"),
        placeholder: S.i32,
        b: S.ptr(dtype, "global"),
    ):
        va = S.vload(a, stride=stride_imm)
        S.vstore(va, b)

    return vload_stride_with_stride_var if is_stride_var else vload_stride_with_stride_imm


def gen_vstore_stride(is_stride_var, dtype, stride_imm):
    @S.prim_func
    def vstore_stride_with_stride_var(
        a: S.ptr(dtype, "global"),
        stride_var: S.i32,
        b: S.ptr(dtype, "global"),
    ):
        va = S.vload(a)
        S.vstore(va, b, stride=stride_var)

    @S.prim_func
    def vstore_stride_with_stride_imm(
        a: S.ptr(dtype, "global"),
        placeholder: S.i32,
        b: S.ptr(dtype, "global"),
    ):
        va = S.vload(a)
        S.vstore(va, b, stride=stride_imm)

    return vstore_stride_with_stride_var if is_stride_var else vstore_stride_with_stride_imm


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vload_vstore(dtype):
    n = hw_native_vdtype(dtype).lanes
    a = np.array(list(range(n)), dtype=dtype)
    gt_out = a

    func = gen_vload_vstore(dtype)
    bm = BuildManager()
    ex = bm.build(func)

    py_out = np.empty(n, dtype)
    func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("is_stride_var", (True, False))
def test_vload_stride(dtype, is_stride_var):
    stride = 3
    n = hw_native_vdtype(dtype).lanes
    a = np.array(list(range(n * stride)), dtype=dtype)
    gt_out = np.array([a[i * stride] for i in range(n)], dtype=dtype)

    func = gen_vload_stride(is_stride_var, dtype, stride)
    bm = BuildManager()
    ex = bm.build(func)

    py_out = np.empty(n, dtype)
    func(a, stride, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, stride, npu_out)
    assert_allclose(npu_out, gt_out)


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("is_stride_var", (True, False))
def test_vstore_stride(dtype, is_stride_var):
    stride = 3
    n = hw_native_vdtype(dtype).lanes
    a = np.array(list(range(n)), dtype=dtype)

    gt_out = np.zeros(n * stride, dtype=dtype)
    indices = np.arange(0, n * stride, stride)
    gt_out[indices] = a

    func = gen_vstore_stride(is_stride_var, dtype, stride)
    bm = BuildManager()
    ex = bm.build(func)

    py_out = np.empty(n * stride, dtype)
    func(a, stride, py_out)
    assert_allclose(py_out[indices], gt_out[indices])

    npu_out = np.empty(n * stride, dtype)
    ex(a, stride, npu_out)
    assert_allclose(npu_out[indices], gt_out[indices])


def gen_vload_vstore_mask(mask, dtype):
    @S.prim_func
    def f_vload_mask(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global")):
        va = S.vload(a, mask=mask)
        S.vstore(va, b)

    @S.prim_func
    def f_vstore_mask(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global")):
        va = S.vload(a)
        S.vstore(va, b, mask=mask)

    return f_vload_mask, f_vstore_mask


def test_vload_vstore_mask():
    n = 8
    dtype = "int32"

    # mask
    mask_str = "3T5F"
    mask_list = [True, True, True, False, False, False, False, False]

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=dtype)
    gt_out = np.array([1, 2, 3, 0, 0, 0, 0, 0], dtype=dtype)

    def check_gt(func, gt_out, assert_mask):
        bm = BuildManager()
        ex = bm.build(func)

        py_out = np.empty(n, dtype)
        func(a, py_out)
        assert_allclose(py_out[assert_mask], gt_out[assert_mask])

        npu_out = np.empty(n, dtype)
        ex(a, npu_out)
        assert_allclose(npu_out[assert_mask], gt_out[assert_mask])

    # func
    f_vload0, f_vstore0 = gen_vload_vstore_mask(mask_str, dtype)
    f_vload1, f_vstore1 = gen_vload_vstore_mask(mask_list, dtype)
    # vload
    check_gt(f_vload0, gt_out, assert_mask=[True] * n)
    check_gt(f_vload1, gt_out, assert_mask=[True] * n)
    # vstore
    check_gt(f_vstore0, gt_out, assert_mask=mask_list)
    check_gt(f_vstore1, gt_out, assert_mask=mask_list)


def gen_vload_vstore_scalar(n, dtype, factor, mask):
    @S.prim_func
    def func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        for i in range(n // factor):
            va = S.vload(a + i * factor)
            vb = S.vload(b + i * factor)
            vc = va + vb
            S.vstore(vc, c + i * factor)

        tail_begin = n // factor * factor
        va = S.vload(a + tail_begin, mask=mask)
        vb = S.vload(b + tail_begin, mask=mask)
        vc = va + vb
        S.vstore(vc, c + tail_begin, mask=mask)

    return func


def test_vload_vstore_scalar():
    n = 35
    dtype = "int8"
    factor = 32
    shape = (n,)
    mask = [True] * 3
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = a + b

    prim_func = gen_vload_vstore_scalar(n, dtype, factor, mask)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(shape, dtype=dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(shape, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_vload_vstore_vector(vdtype, vshape, indices):
    @S.prim_func
    def func(A: S.ptr(vdtype, "global"), B: S.ptr(vdtype, "global"), C: S.ptr(vdtype, "global")):
        a = S.match_buffer(A, vshape)
        b = S.match_buffer(B, vshape)
        c = S.match_buffer(C, vshape)

        va = S.vload(a.addr_of(indices))
        vb = S.vload(b.addr_of(indices))
        vc = va + vb
        S.vstore(vc, c.addr_of(indices))

    return func


def test_vload_vstore_vector():
    n = 3 * 4 * 32
    shape = (3, 4, 32)
    vshape = (3, 4)
    indices = (1, 2)
    dtype = "int8"
    vdtype = hw_native_vdtype(dtype)
    a = rand(n, dtype).reshape(shape)
    b = rand(n, dtype).reshape(shape)
    gt_out = a + b

    prim_func = gen_vload_vstore_vector(vdtype, vshape, indices)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(shape, dtype=dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out[indices], gt_out[indices])

    npu_out = np.empty(shape, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out[indices], gt_out[indices])


def gen_vload_vstore_vector_mask(vdtype):
    @S.prim_func
    def func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        va = S.vload(a, mask="16T16F")
        vb = S.vload(b, mask="16F16T")
        vc = va + vb
        S.vstore(vc, c, mask="16TF")

    return func


def test_vload_vstore_vector_mask():
    n = 32
    dtype = "int8"
    vdtype = hw_native_vdtype(dtype)
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = np.zeros(n, dtype=dtype)
    gt_out[::2] = np.concatenate((a[:16], b[16:]))[::2]

    prim_func = gen_vload_vstore_vector_mask(vdtype)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype=dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out[::2], gt_out[::2])

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out[::2], gt_out[::2])


def gen_vload_vstore_i8x32_from_i8x8(vdtype0, vdtype1):
    @S.prim_func
    def func(a: S.ptr(vdtype0, "global"), b: S.ptr(vdtype0, "global"), c: S.ptr(vdtype1, "global")):
        for i in range(2):
            va = S.vload(a + i * 4, lanes=vdtype1.lanes)
            vb = S.vload(b + i * 4, lanes=vdtype1.lanes)
            vc = va + vb
            S.vstore(vc, c + i)

    return func


def test_vload_vstore_i8x32_from_i8x8():
    n = 64
    dtype = "int8"
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = a + b

    prim_func = gen_vload_vstore_i8x32_from_i8x8(DataType("int8x8"), DataType("int8x32"))
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype=dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vload_vstore("int8")
    test_vload_stride("int8", is_stride_var=True)
    test_vstore_stride("int8", is_stride_var=False)
    test_vload_vstore_mask()
    test_vload_vstore_scalar()
    test_vload_vstore_vector()
    test_vload_vstore_vector_mask()
    test_vload_vstore_i8x32_from_i8x8()
