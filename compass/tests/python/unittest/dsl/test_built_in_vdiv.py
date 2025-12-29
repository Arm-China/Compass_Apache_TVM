# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import get_range
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vdiv(vdtype, mask):
    @S.prim_func
    def vdiv_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        c[0] = S.vdiv(a[0], b[0], mask)

    return vdiv_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float32"))
def test_vdiv(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    mask = rand(n, "bool")
    if vdtype.is_float:
        gt_out = (a / b).astype(dtype)
    else:
        gt_out = np.where(b != 0, np.clip(a / b, *get_range(dtype)), 0).astype(dtype)

    prim_func = gen_vdiv(vdtype, mask)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out[mask], gt_out[mask])

    npu_out = np.empty(n, dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


def test_integer_vdiv_staturate():
    dtype = "int8"
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = np.ones(n, dtype) * (-128)
    b = np.ones(n, dtype) * (-1)

    # The integer vector division instruction "__vdiv" will do saturation.
    gt_out = np.clip(a / b, *get_range(dtype)).astype(dtype)

    prim_func = gen_vdiv(vdtype, mask=None)

    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vdiv("uint32")
    test_integer_vdiv_staturate()
