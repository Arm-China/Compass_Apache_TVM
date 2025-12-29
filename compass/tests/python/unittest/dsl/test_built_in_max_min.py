# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vmax_vmin_imm_func(vdtype, imm):
    @S.prim_func
    def vmax_imm_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global")):
        b[0] = S.max(a[0], imm)

    @S.prim_func
    def vmin_imm_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global")):
        b[0] = S.min(a[0], imm)

    return vmax_imm_func, vmin_imm_func


DTYPE_TUPLE = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32", "bfloat16")


@pytest.mark.parametrize("dtype", DTYPE_TUPLE)
def test_vmax_vmin_imm(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)

    f_vmax, f_vmin = gen_vmax_vmin_imm_func(vdtype, 7)
    bm = BuildManager()

    # vmax
    gt_max = np.array([np.maximum(i, 7) for i in a], dtype=dtype)

    ex = bm.build(f_vmax)

    py_out = np.empty(n, dtype)
    f_vmax(a, py_out)
    assert_allclose(py_out, gt_max)

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_max)

    # vmin
    gt_min = np.array([np.minimum(i, 7) for i in a], dtype=dtype)

    ex = bm.build(f_vmin)

    py_out = np.empty(n, dtype)
    f_vmin(a, py_out)
    assert_allclose(py_out, gt_min)

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_min)


def gen_max_min_func(vdtype):
    @S.prim_func
    def max_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global"), x: S.i8, y: S.i8):
        flag = S.max(x, y)
        if flag == 2:
            c[0] = S.max(a[0], b[0])

    @S.prim_func
    def min_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global"), x: S.i8, y: S.i8):
        flag = S.min(x, y)
        if flag == 1:
            c[0] = S.min(a[0], b[0])

    return max_func, min_func


@pytest.mark.parametrize("dtype", DTYPE_TUPLE)
def test_max_min(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    x, y = 2, 1
    gt_max = np.array([np.maximum(i, j) for i, j in zip(a, b)], dtype=dtype)
    gt_min = np.array([np.minimum(i, j) for i, j in zip(a, b)], dtype=dtype)

    f_max, f_min = gen_max_min_func(vdtype)
    bm = BuildManager()

    # max
    ex = bm.build(f_max)

    py_out = np.empty(n, dtype)
    f_max(a, b, py_out, x, y)
    assert_allclose(py_out, gt_max)

    npu_out = np.empty(n, dtype)
    ex(a, b, npu_out, x, y)
    assert_allclose(npu_out, gt_max)

    # min
    ex = bm.build(f_min)

    py_out = np.empty(n, dtype)
    f_min(a, b, py_out, x, y)
    assert_allclose(py_out, gt_min)

    npu_out = np.empty(n, dtype)
    ex(a, b, npu_out, x, y)
    assert_allclose(npu_out, gt_min)


def test_fail_imm_out_of_range(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func(a: S.ptr("int8x32", "global"), b: S.ptr("int8x32", "global")):
            b[0] = S.max(a[0], 300)

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = 'Can\'t broadcast "300" to any of "(int8x32, uint8x32)".'
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_fail_diff_type(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func(a: S.ptr("i8x32", "global"), b: S.ptr("u8x32", "global"), c: S.ptr("i8x32", "global")):
            c[0] = S.max(a[0], b[0])

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = "The sign of operands is different"
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def max_min_fail(fail_type):
    @S.prim_func
    def fail_max_func(a: S.int8, b: S.int8, c: S.int8):
        c = S.vadd(max(a, b), S.i32x8(1))

    @S.prim_func
    def fail_min_func(a: S.int8, b: S.int8, c: S.int8):
        c = min(a, b)

    return locals()[f"fail_{fail_type}_func"]


@pytest.mark.parametrize("fail_type", ("max", "min"))
def test_fail_invalid_builtin(capfd, fail_type):
    with pytest.raises(RuntimeError):
        BuildManager().build(max_min_fail(fail_type))

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expects = {
        "max": 'The built-in "max" isn\'t supported, please use "S.max".',
        "min": 'The built-in "min" isn\'t supported, please use "S.min".',
    }
    expect = expects[fail_type]
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


if __name__ == "__main__":
    test_vmax_vmin_imm("int8")
    test_max_min("int8")
    test_fail_imm_out_of_range(None)
    test_fail_diff_type(None)
    test_fail_invalid_builtin(None, "max")
    test_vmax_vmin_imm("bfloat16")
    test_max_min("bfloat16")
