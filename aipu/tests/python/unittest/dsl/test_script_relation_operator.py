# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


NAME2SYMBOL = {
    "equal_to": "==",
    "not_equal_to": "!=",
    "greater_than": ">",
    "less_than": "<",
    "greater_than_equal_to": ">=",
    "less_than_equal_to": "<=",
}


def get_gt_out(op, a, b):
    if op == "==":
        out = np.where(a == b, a, b)
    if op == "!=":
        out = np.where(a != b, a, b)
    if op == ">":
        out = np.where(a > b, a, b)
    if op == "<":
        out = np.where(a < b, a, b)
    if op == ">=":
        out = np.where(a >= b, a, b)
    if op == "<=":
        out = np.where(a <= b, a, b)

    return out


def gen_relation_func(a_dtype, b_dtype, out_dtype, is_a_b_scalar, op):
    @S.prim_func
    def func(a: S.ptr(a_dtype, "global"), b: S.ptr(b_dtype, "global"), out: S.ptr(out_dtype, "global")):
        mask_out = a[0] == b[0]
        if op == "==":
            mask_out = a[0] == b[0]
        if op == "!=":
            mask_out = a[0] != b[0]
        if op == ">":
            mask_out = a[0] > b[0]
        if op == "<":
            mask_out = a[0] < b[0]
        if op == ">=":
            mask_out = a[0] >= b[0]
        if op == "<=":
            mask_out = a[0] <= b[0]

        if is_a_b_scalar:
            if mask_out:
                out[0] = a[0]
            else:
                out[0] = b[0]
        else:
            out[0] = S.vsel(a[0], b[0], mask_out)

    return func


@pytest.mark.parametrize("is_b_scalar", (True, False))
@pytest.mark.parametrize("is_a_scalar", (True, False))
@pytest.mark.parametrize("dtype", ("uint8", "int8", "uint16", "int16", "uint32", "int32", "float16", "float32"))
@pytest.mark.parametrize("op_name", NAME2SYMBOL.keys())
def test_relation(dtype, op_name, is_a_scalar, is_b_scalar):
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

    gt_out = get_gt_out(op, a, b)
    prim_func = gen_relation_func(a_dtype, b_dtype, out_dtype, is_a_b_scalar, op)
    ex = aipu.tir.BuildManager().build(prim_func, name=f"{op_name}_{a_dtype}_{b_dtype}")

    py_out = np.empty(out_elems, dtype)
    prim_func(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(out_elems, dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_relation("uint8", "equal_to", True, True)
    test_relation("int16", "not_equal_to", True, False)
    test_relation("uint8", "greater_than", False, True)
    test_relation("uint32", "greater_than_equal_to", False, False)
