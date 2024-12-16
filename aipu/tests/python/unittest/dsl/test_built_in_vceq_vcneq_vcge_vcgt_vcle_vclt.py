# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


name2sdot_table = {
    "vceq": S.vceq,
    "vcneq": S.vcneq,
    "vcge": S.vcge,
    "vcgt": S.vcgt,
    "vcle": S.vcle,
    "vclt": S.vclt,
}


def get_vcmp_gt_out(func_name, dtype, a, b, mask):
    if func_name == "vceq":
        mask_out = np.where(mask, a == b, False)
    elif func_name == "vcneq":
        mask_out = np.where(mask, a != b, False)
    elif func_name == "vcge":
        mask_out = np.where(mask, a >= b, False)
    elif func_name == "vcgt":
        mask_out = np.where(mask, a > b, False)
    elif func_name == "vcle":
        mask_out = np.where(mask, a <= b, False)
    elif func_name == "vclt":
        mask_out = np.where(mask, a < b, False)

    # 'np.where' does implicitly data type ascending from fp16 to fp32 when input
    # scalar value near range border.
    # >>> np.where([True, False], np.array([1,2], dtype="float16"), 65000.0)
    # array([1.0e+00, 6.5e+04], dtype=float32)
    return np.where(mask_out, a, b).astype(dtype)


def gen_vcmp(sdot_func, vdtype, mask):
    @S.prim_func
    def vcmp_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        mask_out = sdot_func(a[0], b[0], mask)
        out[0] = S.vsel(a[0], b[0], mask_out)

    return vcmp_func


@pytest.mark.parametrize("func_name", ("vceq", "vcneq", "vcge", "vcgt", "vcle", "vclt"))
@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vcmp_vector(func_name, dtype):
    sdot_func = name2sdot_table[func_name]
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    mask = rand(n, "bool")
    gt_out = get_vcmp_gt_out(func_name, dtype, a, b, mask)

    f_vcmp = gen_vcmp(sdot_func, vdtype, mask)
    bm = aipu.tir.BuildManager()
    ex = bm.build(f_vcmp, name=f"{func_name}_{dtype}_vector")

    py_out = np.empty(n, dtype)
    f_vcmp(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_vcmp_with_scalar(sdot_func, vdtype, scalar, mask):
    @S.prim_func
    def vcmp_func(v: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        mask_out = sdot_func(v[0], scalar, mask)
        out[0] = S.vsel(v[0], scalar, mask_out)

    return vcmp_func


@pytest.mark.parametrize("func_name", ("vceq", "vcneq", "vcge", "vcgt", "vcle", "vclt"))
@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vcmp_scalar(func_name, dtype):
    sdot_func = name2sdot_table[func_name]
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    scalar = rand(1, dtype, return_python_type=True)
    vector = rand(n, dtype)
    mask = rand(n, "bool")
    gt_out = get_vcmp_gt_out(func_name, dtype, vector, scalar, mask)

    f_vcmp = gen_vcmp_with_scalar(sdot_func, vdtype, scalar, mask)
    bm = aipu.tir.BuildManager()
    ex = bm.build(f_vcmp, name=f"{func_name}_{dtype}_scalar")

    py_out = np.empty(n, dtype)
    f_vcmp(vector, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype)
    ex(vector, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_vcmp_vector(func_name="vcgt", dtype="int8")
    test_vcmp_scalar(func_name="vcgt", dtype="int8")
    test_vcmp_scalar(func_name="vcgt", dtype="float16")
