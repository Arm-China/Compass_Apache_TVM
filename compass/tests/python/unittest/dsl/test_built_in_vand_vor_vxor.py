# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import operator
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


name2sdot_table = {"vand": S.vand, "vor": S.vor, "vxor": S.vxor}
name2operator_table = {"vand": operator.and_, "vor": operator.or_, "vxor": operator.xor}


def gen_bitwise_gentype(sdot_func, vdtype, mask):
    @S.prim_func
    def bitwise_func(x: S.ptr(vdtype, "global"), y: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = sdot_func(x[0], y[0], mask)

    return bitwise_func


def get_bitwise_gt_out(func_name, x, y, mask, is_xy_mask):
    if is_xy_mask:
        mask_x, mask_y = x > 0, y > 0
        if func_name == "vand":
            mask_out = np.where(mask, mask_x & mask_y, False)
        elif func_name == "vor":
            mask_out = np.where(mask, mask_x | mask_y, False)
        elif func_name == "vxor":
            mask_out = np.where(mask, mask_x ^ mask_y, False)
        return np.where(mask_out, x, y)

    if func_name == "vand":
        return np.where(mask, x & y, 0)
    elif func_name == "vor":
        return np.where(mask, x | y, 0)
    return np.where(mask, x ^ y, 0)


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
@pytest.mark.parametrize("func_name", ("vand", "vor", "vxor"))
def test_bitwise_gentype(dtype, func_name):
    sdot_func = name2sdot_table[func_name]
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    x = rand(n, dtype)
    y = rand(n, dtype)
    mask = rand(n, "bool")
    gt_out = get_bitwise_gt_out(func_name, x, y, mask, is_xy_mask=False)

    f_bitwise = gen_bitwise_gentype(sdot_func, vdtype, mask)
    bm = BuildManager()
    ex = bm.build(f_bitwise)

    py_out = np.empty(n, dtype)
    f_bitwise(x, y, py_out)
    assert_allclose(py_out[mask], gt_out[mask])

    npu_out = np.empty(n, dtype)
    ex(x, y, npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


def gen_bitwise_pgentype(sdot_func, vdtype, mask):
    @S.prim_func
    def bitwise_func(x: S.ptr(vdtype, "global"), y: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        mask_x = x[0] > 0
        mask_y = y[0] > 0
        mask_out = sdot_func(mask_x, mask_y, mask)
        out[0] = S.vsel(x[0], y[0], mask_out)

    return bitwise_func


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32"))
@pytest.mark.parametrize("func_name", ("vand", "vor", "vxor"))
def test_bitwise_pgentype(dtype, func_name):
    sdot_func = name2sdot_table[func_name]
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    x = rand(n, dtype)
    y = rand(n, dtype)
    mask = rand(n, "bool")
    gt_out = get_bitwise_gt_out(func_name, x, y, mask, is_xy_mask=True)

    f_bitwise = gen_bitwise_pgentype(sdot_func, vdtype, mask)
    bm = BuildManager()
    ex = bm.build(f_bitwise)

    py_out = np.empty(n, dtype)
    f_bitwise(x, y, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(x, y, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_bitwise_pgentype_autobroadcast(sdot_func, vdtype):
    @S.prim_func
    def bitwise_func(x: S.ptr(vdtype, "global"), y: S.i8, out: S.ptr(vdtype, "global")):
        comparator = S.i8(0)
        mask_out = sdot_func(x[0] > comparator, y > comparator)
        out[0] = S.vsel(x[0], y, mask_out)

    return bitwise_func


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32"))
@pytest.mark.parametrize("func_name", ("vand", "vor", "vxor"))
def test_bitwise_pgentype_autobroadcast(dtype, func_name):
    sdot_func = name2operator_table[func_name]
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    x = rand(n, dtype)
    y = rand(1, "int8", low=-72, high=72)
    y_array = np.array([y] * n)
    gt_out = get_bitwise_gt_out(func_name, x, y_array, mask=[True] * n, is_xy_mask=True)

    f_bitwise = gen_bitwise_pgentype_autobroadcast(sdot_func, vdtype)
    bm = BuildManager()
    ex = bm.build(f_bitwise)

    py_out = np.empty(n, dtype)
    f_bitwise(x, y, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(x, y, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_bitwise_gentype(dtype="int8", func_name="vand")
    test_bitwise_pgentype(dtype="int32", func_name="vor")
    test_bitwise_pgentype_autobroadcast(dtype="int8", func_name="vand")
