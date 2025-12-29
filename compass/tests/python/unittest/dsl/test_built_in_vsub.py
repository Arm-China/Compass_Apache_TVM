# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import get_range
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def get_vsub_gt(dtype, a, b, mask, saturate):
    if saturate:
        assert not dtype.startswith("f"), "Saturate doesn't support float dtype"
        a = np.array(a).astype("int64")
        b = np.array(b).astype("int64")
        out = np.clip(a - b, *get_range(dtype))
    else:
        a = np.array(a, dtype)
        out = np.subtract(a, np.array(b, dtype))
    return np.where(mask, out, a).astype(dtype)


def gen_vsub_gentype_gentype(vdtype, mask, saturate):
    @S.prim_func
    def vsub_a_b_mask(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        c[0] = S.vsub(a[0], b[0], mask, saturate=saturate)

    return vsub_a_b_mask


DTYPE_TUPLE = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32", "bfloat16")


@pytest.mark.parametrize("dtype", DTYPE_TUPLE)
@pytest.mark.parametrize("saturate", (True, False))
@pytest.mark.parametrize("has_mask", (True, False))
def test_vsub_gentype_gentype(dtype, saturate, has_mask):
    if saturate and dtype.startswith(("f", "bf")):
        pytest.skip("Unsupported OpenCL saturate vsub for float precision")

    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    mask = rand(n, "bool") if has_mask else [True] * n
    gt_out = get_vsub_gt(dtype, a, b, mask, saturate)

    prim_func = gen_vsub_gentype_gentype(vdtype, mask, saturate)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype=dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out[mask], gt_out[mask])

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


def gen_vsub_gentype_scalar_imm(vdtype, mask, saturate, imm_b):
    @S.prim_func
    def vsub_a_immb_mask(a: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        vc = S.vsub(a[0], imm_b, mask, saturate=saturate)
        c[0] = S.vsub(imm_b, vc, mask, saturate=saturate)

    return vsub_a_immb_mask


@pytest.mark.parametrize("dtype", DTYPE_TUPLE)
@pytest.mark.parametrize("saturate", (True, False))
@pytest.mark.parametrize("has_mask", (True, False))
def test_vsub_gentype_scalar_imm(dtype, saturate, has_mask):
    if saturate and dtype.startswith(("f", "bf")):
        pytest.skip("Unsupported OpenCL saturate vsub for float precision")

    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(1, dtype, return_python_type=True)
    mask = rand(n, "bool") if has_mask else [True] * n
    gt_tmp = get_vsub_gt(dtype, a, b, mask, saturate)
    gt_out = get_vsub_gt(dtype, b, gt_tmp, mask, saturate)

    prim_func = gen_vsub_gentype_scalar_imm(vdtype, mask, saturate, b)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype=dtype)
    prim_func(a, py_out)
    assert_allclose(py_out[mask], gt_out[mask])

    npu_out = np.empty(n, dtype=dtype)
    ex(a, npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


def gen_vsub_gentype_scalar_var(vdtype, var_dtype, mask, saturate, imm_b):
    @S.prim_func
    def vsub_a_varb_mask(a: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        B = S.cast(imm_b, var_dtype)
        vc = S.vsub(a[0], B, mask, saturate=saturate)
        c[0] = S.vsub(B, vc, mask, saturate=saturate)

    return vsub_a_varb_mask


@pytest.mark.parametrize("dtype", DTYPE_TUPLE)
@pytest.mark.parametrize("saturate", (True, False))
@pytest.mark.parametrize("has_mask", (True, False))
def test_vsub_gentype_scalar_var(dtype, saturate, has_mask):
    if saturate and dtype.startswith(("f", "bf")):
        pytest.skip("Unsupported OpenCL saturate vsub for float precision")

    bm = BuildManager()
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    dtype2var_dtype = {
        "int8": ("int8",),
        "uint8": ("uint8",),
        "int16": ("int16", "int8"),
        "uint16": ("uint16", "uint8"),
        "int32": ("int32", "int16", "int8"),
        "uint32": ("uint32", "uint16", "uint8"),
        "float16": ("float16",),
        "float32": ("float32", "float16"),
        "bfloat16": ("bfloat16", "float16"),
    }
    for var_dtype in dtype2var_dtype[dtype]:
        a = rand(n, dtype)
        b = rand(1, var_dtype, return_python_type=True)
        mask = rand(n, "bool") if has_mask else [True] * n
        gt_tmp = get_vsub_gt(dtype, a, b, mask, saturate)
        gt_out = get_vsub_gt(dtype, b, gt_tmp, mask, saturate)

        prim_func = gen_vsub_gentype_scalar_var(vdtype, var_dtype, mask, saturate, b)
        ex = bm.build(prim_func)

        py_out = np.empty(n, dtype=dtype)
        prim_func(a, py_out)
        assert_allclose(py_out[mask], gt_out[mask])

        npu_out = np.empty(n, dtype=dtype)
        ex(a, npu_out)
        assert_allclose(npu_out[mask], gt_out[mask])


def vsub_fail(fail_type):
    @S.prim_func
    def vsub_fail_diff_dtype(a: S.ptr("u8x32", "global"), b: S.ptr("i16x16", "global"), c: S.ptr("u8x32", "global")):
        c[0] = S.vsub(a[0], b[0])

    @S.prim_func
    def vsub_fail_scalar_imm_out_of_range(a: S.ptr("int8x32", "global"), c: S.ptr("int8x32", "global")):
        c[0] = S.vsub(a[0], 300)

    @S.prim_func
    def vsub_fail_scalar_var_out_of_range(a: S.ptr("int8x32", "global"), B: S.int16, c: S.ptr("int8x32", "global")):
        c[0] = S.vsub(a[0], B)

    if fail_type == "different_dtype":
        return vsub_fail_diff_dtype
    if fail_type == "scalar_imm_out_of_range":
        return vsub_fail_scalar_imm_out_of_range

    assert fail_type == "scalar_var_out_of_range"
    return vsub_fail_scalar_var_out_of_range


@pytest.mark.parametrize("fail_type", ("different_dtype", "scalar_imm_out_of_range", "scalar_var_out_of_range"))
def test_vsub_fail(capfd, fail_type):
    prim_func = vsub_fail(fail_type)
    with pytest.raises(RuntimeError):
        BuildManager().build(prim_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expects = {
        "different_dtype": "Argument type mismatch: ",
        "scalar_imm_out_of_range": 'Can\'t broadcast "300" to any of "(int8x32, uint8x32)"',
        "scalar_var_out_of_range": 'Can\'t broadcast "int16" to any of "(int8x32, uint8x32)"',
    }
    expect = expects[fail_type]
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


if __name__ == "__main__":
    test_vsub_gentype_gentype(dtype="int8", saturate=True, has_mask=True)
    test_vsub_gentype_gentype(dtype="float32", saturate=False, has_mask=True)
    test_vsub_gentype_scalar_imm(dtype="int8", saturate=False, has_mask=False)
    test_vsub_gentype_scalar_var(dtype="uint32", saturate=True, has_mask=True)
    test_vsub_fail(None, "scalar_imm_out_of_range")
    test_vsub_gentype_gentype(dtype="bfloat16", saturate=False, has_mask=True)
    test_vsub_gentype_scalar_imm(dtype="bfloat16", saturate=False, has_mask=False)
    test_vsub_gentype_scalar_var(dtype="bfloat16", saturate=True, has_mask=True)
