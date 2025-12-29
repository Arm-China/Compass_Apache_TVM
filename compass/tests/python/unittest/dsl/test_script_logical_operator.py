# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def get_gt_out(op, a):
    if op == "and":
        out = np.clip(a, 1, 5)
    if op == "or":
        out = np.clip(a, 1, 5)
    if op == "not":
        out = np.where(a != 0, 1, 0).astype(a.dtype)

    return out


def gen_logical_func(a_dtype, out_dtype, op, len_a):
    @S.prim_func
    def func(a: S.ptr(a_dtype, "global"), out: S.ptr(out_dtype, "global")):
        if op == "and":
            for i in range(len_a):
                if a[i] >= 1 and a[i] <= 5:
                    out[i] = a[i]
                elif a[i] < 1:
                    out[i] = 1
                else:
                    out[i] = 5
        if op == "or":
            for i in range(len_a):
                if a[i] < 1 or a[i] > 5:
                    if a[i] < 1:
                        out[i] = 1
                    else:
                        out[i] = 5
                else:
                    out[i] = a[i]
        if op == "not":
            for i in range(len_a):
                if not a[i] == 0:
                    out[i] = 1
                else:
                    out[i] = 0

    return func


@pytest.mark.parametrize("dtype", ("uint8", "int8", "uint16", "int16", "uint32", "int32", "float16", "float32"))
@pytest.mark.parametrize("op", ("and", "or", "not"))
def test_logical(dtype, op):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    va = rand(n, dtype)
    gt_out = get_gt_out(op, va)

    prim_func = gen_logical_func(dtype, dtype, op, n)
    ex = BuildManager().build(prim_func, name=f"Logical_{op}_{vdtype}")

    py_out = np.empty(n, dtype)
    prim_func(va, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(va, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_logical("uint32", "and")
    test_logical("uint8", "or")
    test_logical("uint16", "not")
