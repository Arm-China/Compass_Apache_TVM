# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import get_range, DataType
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


NAME2SYMBOL = {
    "assign": "=",
    "add_assign": "+=",
    "subtract_assign": "-=",
    "multiply_assign": "*=",
    "divide_true_assign": "/=",
    "divide_floor_assign": "//=",
    "mod_assign": "%=",
    "exponent_assign": "**=",
    "bitwise_and_assign": "&=",
    "bitwise_or_assign": "|=",
    "bitwise_xor_assign": "^=",
    "right_shift_assign": ">>=",
    "left_shift_assign": "<<=",
}


def get_gt_out(op, a, b, is_a_scalar):
    out = a.copy()
    if op == "=":
        if len(b) == 1:
            out = [b[0] for i in range(len(a))]
        else:
            out = b
    if op == "+=":
        out += b
    if op == "-=":
        out -= b
    if op == "*=":
        out *= b
    if op == "/=" or op == "//=":
        out = out.astype(np.float64)
        b = b.astype(np.float64)
        out /= b
        if DataType(a.dtype).is_integer:
            out = np.clip(out, *get_range(a.dtype))
        out = out.astype(a.dtype)
    if op == "%=":
        out = np.fmod(out, b).astype(a.dtype)
    if op == "**=":
        out **= b
    if op == "&=":
        out &= b
    if op == "|=":
        out |= b
    if op == "^=":
        out ^= b
    if op == ">>=":
        b = b % 32 if is_a_scalar else b
        out >>= b
    if op == "<<=":
        b = b % 32 if is_a_scalar else b
        out <<= b

    return out


def gen_assignment_func(op, a_dtype, b_dtype, out_dtype):
    @S.prim_func
    def func(a: S.ptr(a_dtype, "global"), b: S.ptr(b_dtype, "global"), out: S.ptr(out_dtype, "global")):
        if S.get_local_id() == 0:
            out[0] = a[0]
            if op == "=":
                out[0] = b[0]
            if op == "+=":
                out[0] += b[0]
            if op == "-=":
                out[0] -= b[0]
            if op == "*=":
                out[0] *= b[0]
            if op == "/=":
                out[0] /= b[0]
            if op == "//=":
                out[0] //= b[0]
            if op == "%=":
                out[0] %= b[0]
            if op == "**=":
                out[0] **= b[0]
            if op == "&=":
                out[0] &= b[0]
            if op == "|=":
                out[0] |= b[0]
            if op == "^=":
                out[0] ^= b[0]
            if op == ">>=":
                out[0] >>= b[0]
            if op == "<<=":
                out[0] <<= b[0]

    return func


def skip_case(dtype, op, out_dtype, is_a_scalar, is_b_scalar):
    if is_a_scalar and not is_b_scalar:
        pytest.skip("Can not assign vector to scalar.")
    if out_dtype.is_bfloat16_scalar:
        pytest.skip("No compute instruction for bfloat16 scalar")

    if op in ("%=", "&=", "|=", "^=", ">>=", "<<=", "//=") and dtype.is_floating:
        pytest.skip(f'Operator "{op}" only supports integer, but got: "{dtype}"')
    if op == "**=":
        msg = f'Operator "{op}" only supports float, but got: "{dtype}"'
        if dtype.is_integer:
            pytest.skip(msg)
        if dtype.is_bfloat16:
            pytest.xfail(msg)
    if op == "*=" and dtype.bits == 8 and out_dtype.is_vector:
        pytest.skip("The 8bit equal-width multiply is meaningless.")
    if op == "/=" and out_dtype.is_floating16_vector:
        pytest.skip("Only support integer and float32 instruction.")


@pytest.mark.parametrize("is_b_scalar", (True, False))
@pytest.mark.parametrize("is_a_scalar", (True, False))
@pytest.mark.parametrize(
    "dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32", "bfloat16")
)
@pytest.mark.parametrize("op_name", NAME2SYMBOL.keys())
def test_assignment(dtype, op_name, is_a_scalar, is_b_scalar):
    dtype = DataType(dtype)
    vdtype = hw_native_vdtype(dtype)
    op = NAME2SYMBOL[op_name]

    n = vdtype.lanes
    va = rand(n, dtype)
    vb = rand(n, dtype)
    a = np.array([va[0]]) if is_a_scalar else va
    a_dtype = vdtype.element_of if is_a_scalar else vdtype
    b = np.array([vb[0]]) if is_b_scalar else vb
    b_dtype = vdtype.element_of if is_b_scalar else vdtype

    is_a_b_scalar = is_a_scalar and is_b_scalar
    out_dtype = dtype if is_a_b_scalar else vdtype
    out_elems = 1 if is_a_b_scalar else n

    skip_case(dtype, op, out_dtype, is_a_scalar, is_b_scalar)

    # avoid divide by zero
    if op in ("/=", "//="):
        b = np.where(b == 0, 1, b)

    gt_out = get_gt_out(op, a, b, is_a_scalar)
    prim_func = gen_assignment_func(op, a_dtype, b_dtype, out_dtype)
    ex = BuildManager().build(prim_func, name=f"{op_name}_{a_dtype}_{b_dtype}")

    py_out = np.empty(out_elems, dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(out_elems, dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_assignment("uint16", "add_assign", False, True)
    test_assignment("uint16", "subtract_assign", False, True)
