# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import get_range
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


DTYPE_TUPLE = ("int8", "int16", "int32", "uint8", "uint16", "uint32", "float16", "float32", "bfloat16")


def get_vrpmax_gt(a, mask, dtype, return_idx=False):
    out = np.zeros(len(a), dtype=dtype)
    out[0] = np.max(a, initial=get_range(dtype)[0], where=mask)
    if return_idx:
        active_indices = np.where(mask)[0]
        active_values = a[active_indices]
        idx = active_indices[np.where(active_values == out[0])[0][0]]
        out[1] = np.array([idx]).view(dtype)[0]
    return out


def get_vrpmin_gt(a, mask, dtype, return_idx=False):
    out = np.zeros(len(a), dtype=dtype)
    out[0] = np.min(a, initial=get_range(dtype)[1], where=mask)
    if return_idx:
        active_indices = np.where(mask)[0]
        active_values = a[active_indices]
        idx = active_indices[np.where(active_values == out[0])[0][0]]
        out[1] = np.array([idx]).view(dtype)[0]
    return out


def vrpmax_gen(vdtype, mask, return_idx=False):
    @S.prim_func
    def vrpmax_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global")):
        b[0] = S.vrpmax(a[0], mask, return_idx)

    return vrpmax_func


def vrpmin_gen(vdtype, mask, return_idx=False):
    @S.prim_func
    def vrpmin_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global")):
        b[0] = S.vrpmin(a[0], mask, return_idx)

    return vrpmin_func


@pytest.mark.parametrize("dtype", DTYPE_TUPLE)
def test_all_vrpmax(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    mask = rand(n, "bool")
    # if all mask is False, skip this test_case
    if not mask.any():
        return
    gt_out = get_vrpmax_gt(a, mask, dtype)

    prim_func = vrpmax_gen(vdtype, mask)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, py_out)
    assert_allclose(py_out[0], gt_out[0])

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out[0], gt_out[0])


@pytest.mark.parametrize("dtype", DTYPE_TUPLE)
def test_all_vrpmin(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    mask = rand(n, "bool")
    # if all mask is False, skip this test_case
    if not mask.any():
        return
    gt_out = get_vrpmin_gt(a, mask, dtype)

    prim_func = vrpmin_gen(vdtype, mask)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, py_out)
    assert_allclose(py_out[0], gt_out[0])

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out[0], gt_out[0])


@pytest.mark.NOT_X1
@pytest.mark.NOT_X2
@pytest.mark.parametrize("dtype", DTYPE_TUPLE)
def test_all_vrpmaxe(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    mask = rand(n, "bool")
    # if all mask is False, skip this test_case
    if not mask.any():
        return
    gt_out = get_vrpmax_gt(a, mask, dtype, return_idx=True)

    prim_func = vrpmax_gen(vdtype, mask, return_idx=True)
    bm = BuildManager(target="X3P_1304")
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, py_out)
    assert_allclose(py_out[:2], gt_out[:2])

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out[:2], gt_out[:2])


@pytest.mark.NOT_X1
@pytest.mark.NOT_X2
@pytest.mark.parametrize("dtype", DTYPE_TUPLE)
def test_all_vrpmine(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = np.array(range(n), dtype=dtype)
    mask = rand(n, "bool")
    # if all mask is False, skip this test_case
    if not mask.any():
        return
    gt_out = get_vrpmin_gt(a, mask, dtype, return_idx=True)

    prim_func = vrpmin_gen(vdtype, mask, return_idx=True)
    bm = BuildManager(target="X3P_1304")
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, py_out)
    assert_allclose(py_out[:2], gt_out[:2])

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out[:2], gt_out[:2])


if __name__ == "__main__":
    test_all_vrpmax("int8")
    test_all_vrpmin("int32")
    test_all_vrpmaxe("int8")
    test_all_vrpmine("float32")
