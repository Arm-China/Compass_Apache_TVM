# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


SYMBOL2NAME = {
    "&": "bitwise_and",
    "|": "bitwise_or",
    "~": "bitwise_not",
    "^": "bitwise_xor",
    "<<": "left_shift",
    ">>": "right_shift",
}
NAME2SYMBOL = {v: k for k, v in SYMBOL2NAME.items()}


def get_gt_out(op, a, b, dtype, is_a_b_scalar):
    if op == "&":
        out = np.bitwise_and(a, b)
    if op == "|":
        out = np.bitwise_or(a, b)
    if op == "~":
        out = np.bitwise_not(a)
    if op == "^":
        out = np.bitwise_xor(a, b)
    if op == "<<":
        b = b % 32 if is_a_b_scalar else b
        out = a << b
    if op == ">>":
        b = b % 32 if is_a_b_scalar else b
        out = a >> b

    return out.astype(dtype)


def gen_bitwise_func(a_dtype, b_dtype, out_dtype, op):
    @S.prim_func
    def func(a: S.ptr(a_dtype, "global"), b: S.ptr(b_dtype, "global"), out: S.ptr(out_dtype, "global")):
        if op == "&":
            out[0] = a[0] & b[0]
        if op == "|":
            out[0] = a[0] | b[0]
        if op == "~":
            out[0] = ~a[0]
        if op == "^":
            out[0] = a[0] ^ b[0]
        if op == "<<":
            out[0] = a[0] << b[0]
        if op == ">>":
            out[0] = a[0] >> b[0]

    return func


@pytest.mark.parametrize("is_b_scalar", (True, False))
@pytest.mark.parametrize("is_a_scalar", (True, False))
@pytest.mark.parametrize("dtype", ("uint8", "int8", "uint16", "int16", "uint32", "int32"))
@pytest.mark.parametrize("op_name", NAME2SYMBOL.keys())
def test_bitwise(dtype, op_name, is_a_scalar, is_b_scalar):
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
    out_dtype = vdtype.element_of if is_a_b_scalar else vdtype
    out_elems = 1 if is_a_b_scalar else n
    if op == "~":
        out_dtype = vdtype.element_of if is_a_scalar else vdtype
        out_elems = 1 if is_a_scalar else n

    gt_out = get_gt_out(op, a, b, dtype, is_a_b_scalar)
    prim_func = gen_bitwise_func(a_dtype, b_dtype, out_dtype, op)
    ex = BuildManager().build(prim_func, name=f"{SYMBOL2NAME[op]}_{a_dtype}_{b_dtype}")

    py_out = np.empty(out_elems, dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(out_elems, dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_bitwise("uint8", "bitwise_and", False, True)
    test_bitwise("int8", "bitwise_or", False, True)
    test_bitwise("uint16", "bitwise_not", True, False)
    test_bitwise("int16", "bitwise_xor", False, True)
    test_bitwise("uint8", "left_shift", False, True)
    test_bitwise("int8", "right_shift", True, False)
