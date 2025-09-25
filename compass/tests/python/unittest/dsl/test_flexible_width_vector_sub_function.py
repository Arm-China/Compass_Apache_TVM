# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


@S.prim_func
def sub_func0(a: S.i32x6, b: S.ptr("int32", "global")):
    S.vstore(a, b)


@S.prim_func
def sub_func1(a: S.i32x9, b: S.ptr("int32", "global")):
    S.vstore(a, b)


@S.prim_func
def sub_func2(a: S.i32x16, b: S.ptr("int32", "global")):
    S.vstore(a, b)


@S.prim_func
def sub_func3(a: S.i32x30, b: S.ptr("int32", "global"), mask: S.boolx30):
    sub_mask_func0(a, b, mask)


@S.prim_func
def sub_mask_func0(a: S.i32x30, b: S.ptr("int32", "global"), mask: S.boolx30):
    sub_mask_func1(a, b, mask)


@S.prim_func
def sub_mask_func1(a: S.i32x30, b: S.ptr("int32", "global"), mask: S.boolx30):
    S.vstore(a, b, mask=mask)


@S.prim_func
def fwv_param_in_sub_func(inp0: S.ptr("int32", "global"), out: S.ptr("int32", "global")):
    va0 = S.vload(inp0, lanes=6)
    va1 = S.vload(inp0 + 6, lanes=9)
    va2 = S.vload(inp0 + 15, lanes=16)
    va3 = S.vload(inp0 + 31, lanes=30)
    sub_func0(va0, out)
    sub_func1(va1, out + 6)
    sub_func2(va2, out + 15)
    sub_func3(va3, out + 31, S.const_mask("25T5F"))


def test_fwv_param_in_sub_func():
    dtype = "int32"
    n = 61
    a = rand(n, dtype)
    gt_out = a.copy()

    bm = BuildManager()
    ex = bm.build(fwv_param_in_sub_func)

    py_out = np.empty(n, dtype=dtype)
    fwv_param_in_sub_func(a, py_out)
    assert_allclose(py_out[:56], gt_out[:56])

    npu_out = np.empty(n, dtype=dtype)
    ex(a, npu_out)
    assert_allclose(npu_out[:56], gt_out[:56])


@S.prim_func
def ret_fwv_subfunc0(a: S.ptr("int32", "global")) -> S.i32x6:
    va = S.vload(a, lanes=6)
    return va


@S.prim_func
def ret_fwv_subfunc1(a: S.ptr("int32", "global")) -> S.i32x9:
    va = S.vload(a, lanes=9)
    return va


@S.prim_func
def ret_fwv_subfunc2(a: S.ptr("int32", "global")) -> S.i32x16:
    va = S.vload(a, lanes=16)
    return va


@S.prim_func
def ret_fwv_subfunc3(a: S.ptr("int32", "global"), b: S.i32) -> S.i32x30:
    va = S.vload(a, lanes=30)
    if b > 0:
        return va
    va += 1
    return va


@S.prim_func
def ret_fwv_func(inp0: S.ptr("int32", "global"), out: S.ptr("int32", "global")):
    va0 = ret_fwv_subfunc0(inp0)
    va1 = ret_fwv_subfunc1(inp0 + 6)
    va2 = ret_fwv_subfunc2(inp0 + 15)
    va3 = ret_fwv_subfunc3(inp0 + 31, 1)
    S.vstore(va0, out)
    S.vstore(va1, out + 6)
    S.vstore(va2, out + 15)
    S.vstore(va3, out + 31)


def test_return_fwv():
    dtype = "int32"
    n = 61
    a = rand(n, dtype)
    gt_out = a

    bm = BuildManager()
    ex = bm.build(ret_fwv_func)

    py_out = np.empty(n, dtype=dtype)
    ret_fwv_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


@S.prim_func
def complex_subfunc0(a: S.ptr("int32", "global")) -> S.i32x16:
    va = S.vload(a, lanes=16)
    return va


@S.prim_func
def indirect_func(a: S.ptr("int32", "global")) -> S.i32x16:
    return complex_subfunc0(a)


@S.prim_func
def complex_subfunc1(va: S.i32x16, vb: S.i32x16) -> S.i32x32:
    return S.vconcat((va, vb))


@S.prim_func
def ret_fwv_complex_func(inp: S.ptr("int32", "global"), out: S.ptr("int32", "global")):
    va0 = indirect_func(inp)
    va1 = S.abs(complex_subfunc0(inp + 16)) + complex_subfunc0(inp + 16)
    va2 = complex_subfunc1(complex_subfunc0(inp + 32), complex_subfunc0(inp + 48))
    vout = S.vconcat([S.vconcat([va0, va1]), va2])
    S.vstore(vout, out)


def test_return_fwv_complex():
    dtype = "int32"
    n = 64
    a = rand(n, dtype)
    gt_out = a.copy()
    gt_out[16:32] += np.abs(gt_out[16:32])

    bm = BuildManager()
    ex = bm.build(ret_fwv_complex_func)

    py_out = np.empty(n, dtype=dtype)
    ret_fwv_complex_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_fwv_param_in_sub_func()
    test_return_fwv()
    test_return_fwv_complex()
