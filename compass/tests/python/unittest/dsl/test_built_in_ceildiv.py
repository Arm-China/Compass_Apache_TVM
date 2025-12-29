# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose
from tvm.compass.dsl.utils import get_binary_op_result_type


def _ceildiv(x, y, out_dtype):
    np_dtype = getattr(np, out_dtype)
    x, y, one = np_dtype(x), np_dtype(y), np_dtype(1)
    return np.trunc((x + y - one) / y).astype(out_dtype)


def gen_ceildiv_func(dtype, divisor_dtype, out_dtype):
    @S.prim_func
    def ceildiv_func(
        a: S.ptr(dtype, "global"), divisors: S.ptr(divisor_dtype, "global"), out: S.ptr(out_dtype, "global")
    ):
        for i in range(8):
            out[i] = S.ceildiv(a[i], divisors[i])

    return ceildiv_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
@pytest.mark.parametrize("divisor_dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_ceildiv(dtype, divisor_dtype):
    n = 8
    a = rand(n, dtype)
    divisors = rand(n, divisor_dtype)
    divisors = np.where(divisors == 0, 1, divisors)
    out_dtype = get_binary_op_result_type(dtype, divisor_dtype)

    gt_out = np.array([_ceildiv(x, y, out_dtype) for x, y in zip(a, divisors)])

    prim_func = gen_ceildiv_func(dtype, divisor_dtype, out_dtype)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, out_dtype)
    prim_func(a, divisors, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, out_dtype)
    ex(a, divisors, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_ceildiv_imm_func(dtype, imm_divisors, out0_dtype, out1_dtype):
    @S.prim_func
    def ceildiv_imm_func(
        a: S.ptr(dtype, "global"), out0: S.ptr(out0_dtype, "global"), out1: S.ptr(out1_dtype, "global")
    ):
        out0[0] = S.ceildiv(a[0], imm_divisors[0])
        out1[0] = S.ceildiv(imm_divisors[1], a[1])

    return ceildiv_imm_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_ceildiv_imm(dtype):
    out_n = 2
    a = rand(out_n, dtype)
    a[1] = 1 if a[1] == 0 else a[1]  # Avoid zero division error
    imm_divisors = [8, 32767]

    out0_dtype = get_binary_op_result_type(dtype, imm_divisors[0])
    out1_dtype = get_binary_op_result_type(imm_divisors[1], dtype)

    gt_out0 = np.array([_ceildiv(a[0], imm_divisors[0], out0_dtype)])
    gt_out1 = np.array([_ceildiv(imm_divisors[1], a[1], out1_dtype)])

    prim_func = gen_ceildiv_imm_func(dtype, imm_divisors, out0_dtype, out1_dtype)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out0, py_out1 = np.empty(1, out0_dtype), np.empty(1, out1_dtype)
    prim_func(a, py_out0, py_out1)
    assert_allclose(py_out0, gt_out0)
    assert_allclose(py_out1, gt_out1)

    npu_out0, npu_out1 = np.empty(1, out0_dtype), np.empty(1, out1_dtype)
    ex(a, npu_out0, npu_out1)
    assert_allclose(npu_out0, gt_out0)
    assert_allclose(npu_out1, gt_out1)


def test_fail_zero_division():
    @S.prim_func
    def func(a: S.ptr("int8", "global"), b: S.ptr("int8", "global")):
        b[0] = S.ceildiv(1, a[0])

    BuildManager().build(func)

    with pytest.raises(ZeroDivisionError) as exc_info:
        func(np.array((0,), "int8"), np.empty(1, "int8"))

    exc_msg = str(exc_info.value)
    expect = "The 2nd arg can't be zero."
    assert expect in exc_msg, f"\nExpect snippet:\n{expect}\n\nException Message:\n{exc_msg}\n"


if __name__ == "__main__":
    test_ceildiv("int8", "int16")
    test_ceildiv_imm("int8")
