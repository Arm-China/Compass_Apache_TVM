# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose

dtype = "int32"
vdtype = "int32x8"


def gen_cmp(cmp_type):
    @S.prim_func
    def gt_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        mask = a[0] > 0
        c[0] = S.vsel(a[0], b[0], mask)

    @S.prim_func
    def ge_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        mask = a[0] >= 0
        c[0] = S.vsel(a[0], b[0], mask)

    @S.prim_func
    def lt_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        mask = a[0] < 0
        c[0] = S.vsel(a[0], b[0], mask)

    @S.prim_func
    def le_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        mask = a[0] <= 0
        c[0] = S.vsel(a[0], b[0], mask)

    @S.prim_func
    def eq_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        mask = a[0] == 0
        c[0] = S.vsel(a[0], b[0], mask)

    @S.prim_func
    def neq_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        mask = a[0] != 0
        c[0] = S.vsel(a[0], b[0], mask)

    return locals()[f"{cmp_type}_func"]


def get_gt(cmp_type, a, b):
    if cmp_type == "gt":
        return np.where(a > 0, a, b)
    if cmp_type == "ge":
        return np.where(a >= 0, a, b)
    if cmp_type == "lt":
        return np.where(a < 0, a, b)
    if cmp_type == "le":
        return np.where(a <= 0, a, b)
    if cmp_type == "eq":
        return np.where(a == 0, a, b)
    assert cmp_type == "neq"
    return np.where(a != 0, a, b)


@pytest.mark.parametrize("cmp_type", ("eq", "neq", "ge", "gt", "le", "lt"))
def test_cmp(cmp_type):
    n = 8
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = get_gt(cmp_type, a, b)

    py_func = gen_cmp(cmp_type)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype)
    py_func(a, b, py_out)
    assert_allclose(gt_out, py_out)

    npu_out = np.empty(n, dtype)
    ex(a, b, npu_out)
    assert_allclose(gt_out, npu_out)


if __name__ == "__main__":
    test_cmp("eq")
