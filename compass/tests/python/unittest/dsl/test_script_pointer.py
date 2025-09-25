# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_pointer_func(dtype):
    vdtype = hw_native_vdtype(dtype)

    @S.prim_func
    def add_by_pointer(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        va = a[:8]
        vb = b[:8]
        vc_ptr = c.as_ptr(vdtype)
        vc_ptr[0] = va + vb

        high_half_ptr = (vc_ptr + 1).as_ptr(dtype)
        for i in range(8):
            high_half_ptr[i] = a[i + 8] + b[i + 8]

    @S.prim_func
    def pointer_func(a: S.ptr(dtype, "global"), b: S.ptr("void", "global"), c: S.ptr(dtype, "global")):
        B = S.match_buffer(b.as_ptr(dtype), (8, 8))

        tid = S.get_local_id()

        tec_a = a + tid * 16
        tec_a = tid * 16 + a
        tec_b = B.addr_of(tid)
        tec_b += tid * 8
        tec_c = c + tid * 16
        add_by_pointer(tec_a, tec_b, tec_c)

    return pointer_func


def test_pointer():
    n = 64
    dtype = "float32"
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = a + b

    pointer_func = gen_pointer_func(dtype)
    ex = BuildManager().build(pointer_func)

    py_out = np.empty(n, dtype)
    pointer_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex.run(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


def ptr_array_overflow(dtype):
    @S.prim_func
    def pointer_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        c[8] = a[7] + b[7]

    return pointer_func


def test_pointer_arry_overflow():
    n = 8
    dtype = "float32"
    a = rand(n, dtype)
    b = rand(n, dtype)

    pointer_func = ptr_array_overflow(dtype)
    with pytest.raises(IndexError):
        _ = BuildManager().build(pointer_func)

        py_out = np.empty(n, dtype)
        pointer_func(a, b, py_out)


def ptr_addr_overflow():
    @S.prim_func
    def pointer_func(c: S.ptr("float32", "global")):
        for i in range(9):
            c[0] = i + 10
            c += 1

    return pointer_func


def test_pointer_addr_overflow():
    pointer_func = ptr_addr_overflow()
    with pytest.raises(AssertionError):
        _ = BuildManager().build(pointer_func)

        py_out = np.empty(8, "float32")
        pointer_func(py_out)


def gen_ptr_compare_func(dtype):
    @S.prim_func
    def ptr_compare_func(src: S.ptr(dtype, "global"), dst: S.ptr(dtype, "global")):
        dst[0:8] = 2

        a = src + 3
        if a > src:
            dst[0] = 1
        if a >= src:
            dst[1] = 1

        b = src + 1
        if b < a:
            dst[2] = 1
        if b <= a:
            dst[3] = 1

        c = b + 2
        if c == a:
            dst[4] = 1
        if c != a:
            dst[5] = 1

    return ptr_compare_func


def test_pointer_compare():
    n = 8
    dtype = "int32"
    a = rand(n, dtype)
    gt_out = np.array([1, 1, 1, 1, 1, 2, 2, 2], dtype)

    py_func = gen_ptr_compare_func(dtype)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_nullptr_func(dtype):
    @S.prim_func
    def nullptr_func(
        inp0: S.ptr(dtype, "global"),
        inp1: S.ptr(dtype, "global"),
        out0: S.ptr(dtype, "global"),
        out1: S.ptr(dtype, "global"),
    ):
        out0[0:8] = 2
        if inp0.is_nullptr:
            out0[0] = S.i32(1)
        else:
            out1[0] = S.i32(1)

        if inp1.is_nullptr:
            out0[1] = S.i32(1)

        if out1.is_nullptr:
            out0[2] = S.i32(1)

    return nullptr_func


def test_nullptr():
    n = 8
    dtype = "int32"
    inp0 = None
    inp1 = rand(n, dtype)
    out1 = 0
    gt_out = np.array([1, 2, 1, 2, 2, 2, 2, 2], dtype)

    py_func = gen_nullptr_func(dtype)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out0 = np.empty(n, dtype=dtype)
    py_func(inp0, inp1, py_out0, out1)
    assert_allclose(py_out0, gt_out)

    npu_out0 = np.empty(n, dtype=dtype)
    ex(inp0, inp1, npu_out0, out1)
    assert_allclose(npu_out0, gt_out)


def gen_same_annotation_func(dtype):
    ptr_dtype_global = S.ptr(hw_native_vdtype(dtype), "global")

    @S.prim_func
    def same_annotation_func(a: ptr_dtype_global, out: ptr_dtype_global):
        out[0] = a[0] + 1

    return same_annotation_func


def test_same_annotation():
    n, dtype = 8, "int32"
    a = rand(n, dtype)
    gt_out = a + 1

    py_func = gen_same_annotation_func(dtype)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_return_ptr_func():
    dtype, vdtype = "i32", "i32x3"

    @S.prim_func
    def ptr_move0(x: S.ptr(dtype, "global")) -> S.ptr(dtype, "global"):
        return x + 1

    @S.prim_func
    def ptr_move1(x: S.ptr(dtype, "global")) -> S.ptr(vdtype, "global"):
        return (x + 3).as_ptr(vdtype)

    @S.prim_func
    def return_ptr_func(a: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        out[0] = a[0] + 1
        cur_out = ptr_move0(out)
        cur_out[0] = a[1] + 1

        cur_out = ptr_move0(cur_out)
        cur_out[0] = a[2] + 1

        i32x3_out = ptr_move1(out)
        i32x3_out[0] = a[3:6] + 1

        if S.get_local_id() == 0:
            sram = S.alloc_buffer([2], dtype, "lsram")
            S.dma_copy(sram, a + 6, 2)
            S.dma_copy(ptr_move1(i32x3_out).as_ptr(dtype), sram, 2)
            out[6] += 1
            out[7] += 1

    return return_ptr_func


def test_return_ptr():
    dtype, n = "int32", 8
    a = rand(n, dtype)
    gt_out = a + 1

    prim_func = gen_return_ptr_func()
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


@S.prim_func
def pointer_reinterpret_integer_func(inp: S.ptr("u32", "global"), out: S.ptr("u32", "global")):
    if S.get_local_id() != 0:
        return

    out[0:8] = inp[0:8]
    out_u32 = S.reinterpret(out, "uint32")
    out_u32_offset_3 = out_u32 + 3 * 4
    out_u32_offset_3_ptr = S.reinterpret(out_u32_offset_3, "uint32 *")
    out_u32_offset_3_ptr[0] += 1


def test_pointer_reinterpret_integer():
    dtype, n = "uint32", 8
    a = rand(n, dtype)
    gt_out = a.copy()
    gt_out[3] += 1

    prim_func = pointer_reinterpret_integer_func
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_pointer()
    test_pointer_arry_overflow()
    test_pointer_addr_overflow()
    test_pointer_compare()
    test_nullptr()
    test_same_annotation()
    test_return_ptr()
    test_pointer_reinterpret_integer()
