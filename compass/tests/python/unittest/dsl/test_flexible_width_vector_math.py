# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from scipy import special
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


name2sdot_unary_table = {
    "tanh": S.tanh,
    "log": S.log,
    "exp": S.exp,
    "rint": S.rint,
    "abs": S.abs,
    "sin": S.sin,
    "cos": S.cos,
    "rsqrt": S.rsqrt,
    "sqrt": S.sqrt,
    "floor": S.floor,
    "ceil": S.ceil,
    "erf": S.erf,
    "exp2": S.exp2,
}


def get_unary_gt_out(func_name, a):
    if func_name == "rsqrt":
        one = np.float32(1)
        neg_zero = np.float32(-0.0)
        mask = np.logical_or(np.isneginf(a), a == neg_zero)
        out = np.where(mask, one / np.sqrt(a), np.sqrt(one / a))
        return out.astype(a.dtype)
    if func_name == "erf":
        np_unary_func = special.erf
    else:
        np_unary_func = getattr(np, func_name)
    return np_unary_func(a).astype(a.dtype)


def gen_unary_func(sdot_func, dtype, n):
    @S.prim_func
    def unary_func(a: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        out[0:n] = sdot_func(a[0:n])

    return unary_func


@pytest.mark.parametrize("dtype", ("float16", "float32", "bfloat16"))
@pytest.mark.parametrize("func_name", name2sdot_unary_table.keys())
def test_unary(func_name, dtype):
    sdot_func = name2sdot_unary_table[func_name]
    n = hw_native_vdtype(dtype).lanes + 1
    a = rand(n, dtype)
    gt_out = get_unary_gt_out(func_name, a)

    py_func = gen_unary_func(sdot_func, dtype, n)
    bm = BuildManager()
    ex = bm.build(py_func, name=f"{func_name}_{dtype}")

    py_out = np.empty(n, dtype)
    py_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_pow_func(n, lanes, dtype):
    @S.prim_func
    def pow_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        c[0:lanes] = a[0:lanes] ** b[0:lanes]
        c[lanes:n] = S.pow(a[lanes:n], b[lanes:n])

    return pow_func


@pytest.mark.parametrize("dtype", ("float16", "float32", "bfloat16"))
@pytest.mark.parametrize("func_name", ("pow",))
def test_binary(func_name, dtype):
    lanes = hw_native_vdtype(dtype).lanes + 3
    n = lanes * 2
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = np.power(a, b).astype(dtype)

    py_func = gen_pow_func(n, lanes, dtype)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_unary("exp", "float32")
    test_unary("tanh", "float32")
    test_unary("log", "float32")
    test_unary("rint", "float32")
    test_unary("abs", "float32")
    test_binary("pow", "float32")
