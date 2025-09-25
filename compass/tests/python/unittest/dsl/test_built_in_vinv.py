# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vinv_gentype(vdtype, mask):
    @S.prim_func
    def vinv_func(x: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vinv(x[0], mask)

    return vinv_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_vinv_gentype(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    x = rand(n, dtype)
    mask = rand(n, "bool")
    gt_out = np.where(mask, ~x, 0)

    f_vinv = gen_vinv_gentype(vdtype, mask)
    bm = BuildManager()
    ex = bm.build(f_vinv, name=f"vinv_gentype_{dtype}")

    py_out = np.empty(n, dtype)
    f_vinv(x, py_out)
    assert_allclose(py_out[mask], gt_out[mask])

    npu_out = np.empty(n, dtype)
    ex(x, npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


def gen_vinv_pgentype(vdtype, mask):
    @S.prim_func
    def vinv_func(x: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        mask_x = x[0] > 0
        mask_out = S.vinv(mask_x, mask)

        out[0] = S.vsel(x[0], 0, mask_out)

    return vinv_func


def get_gt_out_pgentype(x, mask):
    mask_x = x > 0
    mask_out = np.where(mask, ~mask_x, False)  # inactive set 0, False for pgentype
    return np.where(mask_out, x, 0)  # This line equals vsel(x[0], 0, mask_out) in prim func


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32"))
def test_vinv_pgentype(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    x = rand(n, dtype)
    mask = rand(n, "bool")
    gt_out = get_gt_out_pgentype(x, mask)

    f_vinv = gen_vinv_pgentype(vdtype, mask)
    bm = BuildManager()
    ex = bm.build(f_vinv, name=f"vinv_pgentype_{dtype}")

    py_out = np.empty(n, dtype)
    f_vinv(x, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(x, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vinv_gentype("int8")
    test_vinv_pgentype("int8")
