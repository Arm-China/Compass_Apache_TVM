# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def const_mask_generic(vdtype, mask_array):
    @S.prim_func
    def const_mask_list(inp0: S.ptr(vdtype, "global"), inp1: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        mask = S.const_mask(mask_array)
        out[0] = S.vadd(inp0[0], inp1[0], mask=mask)

    return const_mask_list


@pytest.mark.parametrize("mask_type", ("list", "tuple", "np_ndarray"))
def test_const_mask(mask_type):
    dtype = "int32"
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)

    gt_out = np.zeros(n, dtype=dtype)
    gt_out[4:] = a[4:] + b[4:]

    mask = [False, False, False, False, True, True, True, True]
    if mask_type == "tuple":
        mask = tuple(mask)
    elif mask_type == "np_ndarray":
        mask = np.array(mask)

    py_func = const_mask_generic(vdtype, mask)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    assert_allclose(py_out[4:], gt_out[4:])

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out[4:], gt_out[4:])


def const_mask_reassign_generic(vdtype):
    @S.prim_func
    def const_mask_reassign_func(
        inp0: S.ptr(vdtype, "global"),
        inp1: S.ptr(vdtype, "global"),
        out: S.ptr(vdtype, "global"),
        flag: S.i32,
    ):
        mask = S.const_mask([True] * 4 + [False] * 4)
        if flag == 1:
            mask = S.const_mask([False] * 4 + [True] * 4)
        out[0] = S.vadd(inp0[0], inp1[0], mask=mask)

    return const_mask_reassign_func


def test_const_mask_reassign():
    dtype = "int32"
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)

    gt_out = np.zeros(n, dtype=dtype)
    gt_out[4:] = a[4:] + b[4:]

    py_func = const_mask_reassign_generic(vdtype)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out, 1)
    assert_allclose(py_out[4:], gt_out[4:])

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out, 1)
    assert_allclose(npu_out[4:], gt_out[4:])


def gen_vand_pgentype(vdtype, imm_mask0_str, imm_mask1_list):
    @S.prim_func
    def vand_func(x: S.ptr(vdtype, "global"), y: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        imm_mask0 = S.const_mask(imm_mask0_str)
        imm_mask1 = S.const_mask(imm_mask1_list)

        mask_out = S.vand(imm_mask0, imm_mask1)
        out[0] = S.vsel(x[0], y[0], mask_out)

    return vand_func


def test_const_mask_arg():
    dtype = "int32"
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    x = rand(n, dtype)
    y = rand(n, dtype)

    mask0 = np.array([True] * 3 + [False] * 5, "bool")
    mask1 = rand(n, "bool")
    imm_mask0 = "3T5F"
    imm_mask1 = mask1.tolist()
    gt_out = np.where(mask0 & mask1, x, y)

    py_func = gen_vand_pgentype(vdtype, imm_mask0, imm_mask1)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype)
    py_func(x, y, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(x, y, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_const_mask("list")
    test_const_mask_reassign()
    test_const_mask_arg()
