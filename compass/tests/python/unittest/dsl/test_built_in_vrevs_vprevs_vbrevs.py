# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vrevs(vdtype):
    @S.prim_func
    def vrevs_func(x: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vrevs(x[0])

    return vrevs_func


@pytest.mark.parametrize(
    "dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32", "bfloat16")
)
def test_vrevs(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    x = rand(n, dtype)
    gt_out = np.flip(x)

    f_vrevs = gen_vrevs(vdtype)
    bm = BuildManager()
    ex = bm.build(f_vrevs)

    py_out = np.empty(n, dtype)
    f_vrevs(x, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(x, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_vprevs(vdtype):
    @S.prim_func
    def vprevs_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        mask_a = a[0] > 0
        mask_out = S.vrevs(mask_a)
        out[0] = S.vsel(a[0], b[0], mask_out)

    return vprevs_func


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32"))
def test_vprevs(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = np.where(np.flip(a > 0), a, b)

    f_vprevs = gen_vprevs(vdtype)
    bm = BuildManager()
    ex = bm.build(f_vprevs)

    py_out = np.empty(n, dtype)
    f_vprevs(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_vbrevs(vdtype):
    @S.prim_func
    def vbrevs_func(x: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vbrevs(x[0])

    return vbrevs_func


def get_vbrevs_gt_out(x):
    all_bits = np.unpackbits(x.view(np.uint8))  # x0_b x1_b x2_b
    flip_all_bits = np.flip(all_bits)  # rev_x2_b rev_x1_b rev_x0_b
    temp_out = np.packbits(flip_all_bits).view(x.dtype)  # x2_out x1_out x0_out
    return np.flip(temp_out)  # x0_out x1_out x2_out


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_vbrevs(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    x = rand(n, dtype)
    gt_out = get_vbrevs_gt_out(x)

    f_vbrevs = gen_vbrevs(vdtype)
    bm = BuildManager()
    ex = bm.build(f_vbrevs)

    py_out = np.empty(n, dtype)
    f_vbrevs(x, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(x, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vrevs("int8")
    test_vrevs("bfloat16")
    test_vprevs("int32")
    test_vbrevs("int16")
