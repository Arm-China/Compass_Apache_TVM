# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from scipy import special
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


name2sdot_table = {
    "tanh": S.tanh,
    "sin": S.sin,
    "cos": S.cos,
    "floor": S.floor,
    "ceil": S.ceil,
    "sqrt": S.sqrt,
    "rsqrt": S.rsqrt,
    "log": S.log,
    "exp": S.exp,
    "erf": S.erf,
    "exp2": S.exp2,
}


def get_gt_out(func_name, a):
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


def gen_unary_func(sdot_func, buf_dtype, times):
    @S.prim_func
    def unary_func(a: S.ptr(buf_dtype, "global"), out: S.ptr(buf_dtype, "global")):
        for i in range(times):
            out[i] = sdot_func(a[i])

    return unary_func


@pytest.mark.parametrize("func_name", name2sdot_table.keys())
@pytest.mark.parametrize("dtype", ("float16", "float32", "bfloat16"))
def test_math_unary(func_name, dtype):
    sdot_func = name2sdot_table[func_name]
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes

    a = rand(n, dtype)
    gt_out = get_gt_out(func_name, a)

    # vector
    f_unary = gen_unary_func(sdot_func, vdtype, times=1)
    bm = BuildManager()
    ex = bm.build(f_unary, name=f"{func_name}_{dtype}_vector")

    py_out = np.empty(n, dtype)
    f_unary(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)

    if dtype == "bfloat16":
        return  # Scalar built-in not supports blfoat16.

    # scalar
    f_unary = gen_unary_func(sdot_func, dtype, times=n)
    bm = BuildManager()
    ex = bm.build(f_unary, name=f"{func_name}_{dtype}_scalar")

    py_out = np.empty(n, dtype)
    f_unary(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


def test_rsqrt_corner_case():
    func_name, dtype = "rsqrt", "float32"
    sdot_func = name2sdot_table[func_name]

    special_fp32s = [np.uint32(x).view("float32") for x in (0x4B9A8DA2, 0x4B759FCC, 0x4B8D172C, 0x4B8ADD9F)]
    corners = ["-inf", "inf", "nan", 0.0, -0.0, 10.0, -10.0]
    a = np.array(special_fp32s + corners, dtype)
    n = a.size

    gt_out = get_gt_out(func_name, a)
    f_unary = gen_unary_func(sdot_func, dtype, times=n)
    ex = BuildManager().build(f_unary)

    py_out = np.empty(n, dtype)
    f_unary(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)

    for i in range(len(special_fp32s)):
        py_out_u32 = py_out[i].view("uint32")
        npu_out_u32 = npu_out[i].view("uint32")
        assert py_out_u32 == npu_out_u32, f"Binary should be same, but got: {py_out_u32} vs. {npu_out_u32}"


if __name__ == "__main__":
    test_math_unary(func_name="sqrt", dtype="float16")
    test_math_unary(func_name="floor", dtype="float32")
    test_math_unary(func_name="erf", dtype="float32")
    test_rsqrt_corner_case()
    test_math_unary(func_name="tanh", dtype="bfloat16")
    test_math_unary(func_name="sin", dtype="bfloat16")
    test_math_unary(func_name="cos", dtype="bfloat16")
    test_math_unary(func_name="floor", dtype="bfloat16")
    test_math_unary(func_name="ceil", dtype="bfloat16")
    test_math_unary(func_name="sqrt", dtype="bfloat16")
    test_math_unary(func_name="rsqrt", dtype="bfloat16")
    test_math_unary(func_name="log", dtype="bfloat16")
    test_math_unary(func_name="exp", dtype="bfloat16")
    test_math_unary(func_name="erf", dtype="bfloat16")
    test_math_unary(func_name="exp2", dtype="bfloat16")
