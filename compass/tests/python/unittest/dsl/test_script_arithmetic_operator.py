# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import get_range, DataType
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


SYMBOL2NAME = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "truediv",
    "//": "floordiv",
    "%": "mod",
    "**": "pow",
}
NAME2SYMBOL = {v: k for k, v in SYMBOL2NAME.items()}


def get_gt_out(op, a, b, dtype, is_vector):
    if op == "%":
        return np.fmod(a, b).astype(dtype)

    if op == "//":
        op = "/"
    out = eval(f"a {op} b")

    if op == "/":
        if is_vector and dtype.is_integer:
            out = np.clip(out, *get_range(dtype))
        out = out.astype(dtype)
    return out


def gen_arithmetic_func(a_dtype, b_dtype, out_dtype, op):
    @S.prim_func
    def func(a: S.ptr(a_dtype, "global"), b: S.ptr(b_dtype, "global"), out: S.ptr(out_dtype, "global")):
        if op == "+":
            out[0] = a[0] + b[0]
        if op == "-":
            out[0] = a[0] - b[0]
        if op == "*":
            out[0] = a[0] * b[0]
        if op == "/":
            out[0] = a[0] / b[0]
        if op == "//":
            out[0] = a[0] // b[0]
        if op == "%":
            out[0] = a[0] % b[0]
        if op == "**":
            out[0] = a[0] ** b[0]

    return func


def skip_case(dtype, op, out_dtype):
    if out_dtype.is_bfloat16_scalar:
        pytest.skip("No compute instruction for bfloat16 scalar")

    if op in ("%", "//") and dtype.is_floating:
        pytest.skip(f'Operator "{SYMBOL2NAME[op]}" only supports integer, but got: "{dtype}"')
    if op == "**":
        msg = f'Operator "{SYMBOL2NAME[op]}" only supports float, but got: "{dtype}"'
        if dtype.is_integer:
            pytest.skip(msg)
        if dtype.is_bfloat16:
            pytest.xfail(msg)
    if op == "*" and dtype.bits == 8 and out_dtype.is_vector:
        pytest.skip("The 8bit equal-width multiply is meaningless.")
    if op == "/" and out_dtype.is_floating16_vector:
        pytest.skip("Only support integer and float32 instruction.")


@pytest.mark.parametrize("is_b_scalar", (True, False))
@pytest.mark.parametrize("is_a_scalar", (True, False))
@pytest.mark.parametrize(
    "dtype", ("uint8", "int8", "uint16", "int16", "uint32", "int32", "float16", "float32", "bfloat16")
)
@pytest.mark.parametrize("op_name", NAME2SYMBOL.keys())
def test_arithmetic(dtype, op_name, is_a_scalar, is_b_scalar):
    dtype = DataType(dtype)
    vdtype = hw_native_vdtype(dtype)
    op = NAME2SYMBOL[op_name]

    n = vdtype.lanes
    va = rand(n, dtype)
    vb = rand(n, dtype)
    a = np.array([va[0]]) if is_a_scalar else va
    a_dtype = dtype if is_a_scalar else vdtype
    b = np.array([vb[0]]) if is_b_scalar else vb
    b_dtype = dtype if is_b_scalar else vdtype

    is_a_b_scalar = is_a_scalar and is_b_scalar
    out_dtype = dtype if is_a_b_scalar else vdtype
    out_elems = 1 if is_a_b_scalar else n

    skip_case(dtype, op, out_dtype)

    # avoid divide by zero
    if op in ("/", "//"):
        b = np.where(b == 0, 1, b)

    gt_out = get_gt_out(op, a, b, dtype, not is_a_b_scalar)
    prim_func_name = f"{SYMBOL2NAME[op]}_{a_dtype}_{b_dtype}"

    prim_func = gen_arithmetic_func(a_dtype, b_dtype, out_dtype, op)
    bm = BuildManager()
    ex = bm.build(prim_func, name=prim_func_name)

    if not is_a_scalar and not is_b_scalar and op_name in ("add", "sub", "mul"):
        expect = f"a[0] {op} b[0]"
        assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nCompass C code:\n{ex.c_code}\n"

    py_out = np.empty(out_elems, dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(out_elems, dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_arithmetic("float32", "pow", True, True)
    test_arithmetic("float32", "truediv", False, False)
    test_arithmetic("uint16", "add", False, False)
    test_arithmetic("uint16", "sub", False, False)
    test_arithmetic("uint16", "mul", False, False)
    test_arithmetic("bfloat16", "sub", False, True)
    test_arithmetic("bfloat16", "mul", False, False)
    test_arithmetic("bfloat16", "add", False, True)
