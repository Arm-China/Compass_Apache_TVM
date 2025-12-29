# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vsel_gentype(vdtype):
    @S.prim_func
    def vsel_gentype_func(x: S.ptr(vdtype, "global"), y: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        mask = x[0] > 0
        out[0] = S.vsel(x[0], y[0], mask)

    return vsel_gentype_func


@pytest.mark.parametrize(
    "dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32", "bfloat16")
)
def test_vsel_gentype(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    x = rand(n, dtype)
    y = rand(n, dtype)
    gt_out = np.where(x > 0, x, y)

    f_vsel = gen_vsel_gentype(vdtype)
    bm = BuildManager()
    ex = bm.build(f_vsel)

    py_out = np.empty(n, dtype)
    f_vsel(x, y, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(x, y, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_vsel_pgentype(vdtype):
    @S.prim_func
    def vsel_pgentype_func(x: S.ptr(vdtype, "global"), y: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        mask_x = x[0] > 0
        mask_y = y[0] > 0
        mask_out = S.vsel(x=mask_x, y=mask_y, mask=mask_x)

        out[0] = S.vsel(x[0], y[0], mask_out)

    return vsel_pgentype_func


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32"))
def test_vsel_pgentype(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    x = rand(n, dtype)
    y = rand(n, dtype)
    gt_out = np.where(x > 0, x > 0, y > 0)  # 1st vsel
    gt_out = np.where(gt_out, x, y)  # 2nd vsel

    f_vsel = gen_vsel_pgentype(vdtype)
    bm = BuildManager()
    ex = bm.build(f_vsel)

    py_out = np.empty(n, dtype)
    f_vsel(x, y, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(x, y, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vsel_gentype("int8")
    test_vsel_gentype("float32")
    test_vsel_pgentype("int8")
