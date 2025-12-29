# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import get_range
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def get_vadd_gt(dtype, a, b, mask, saturate, out_sign=None):
    if not dtype.startswith("f") and out_sign == "u":
        dtype = f"u{dtype}"

    if saturate:
        assert not dtype.startswith("f"), "Saturate doesn't support float dtype"
        a = np.array(a).astype("int64")
        b = np.array(b).astype("int64")
        out = np.clip(a + b, *get_range(dtype)).astype(dtype)
    else:
        out = np.add(a, np.array(b, dtype))
    return np.where(mask, out * 2, a).astype(dtype)


def vadd_gentype_gentype(vdtype, mask, saturate):
    @S.prim_func
    def vadd_a_b_mask(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        vc = S.vadd(a[0], b[0], mask, saturate=saturate)
        c[0] = vc + S.vadd(b[0], a[0], mask, saturate=saturate)

    return vadd_a_b_mask


DTYPE_TUPLE = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32", "bfloat16")


@pytest.mark.parametrize("dtype", DTYPE_TUPLE)
@pytest.mark.parametrize("saturate", (True, False))
@pytest.mark.parametrize("has_mask", (True, False))
def test_vadd_gentype_gentype(dtype, saturate, has_mask):
    if saturate and dtype.startswith(("f", "bf")):
        pytest.skip("Unsupported OpenCL saturate vadd for float precision")

    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    mask = rand(n, "bool") if has_mask else [True] * n
    gt_out = get_vadd_gt(dtype, a, b, mask, saturate)

    prim_func = vadd_gentype_gentype(vdtype, mask, saturate)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype=dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out[mask], gt_out[mask])

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


def gen_vadd_gentype_scalar_imm(vdtype, mask, saturate, imm_b):
    @S.prim_func
    def vadd_gentype_scalar_imm(a: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        vc = S.vadd(a[0], imm_b, mask, saturate=saturate)
        c[0] = vc + S.vadd(imm_b, a[0], mask, saturate=saturate)

    return vadd_gentype_scalar_imm


@pytest.mark.parametrize("dtype", DTYPE_TUPLE)
@pytest.mark.parametrize("saturate", (True, False))
@pytest.mark.parametrize("has_mask", (True, False))
def test_vadd_gentype_scalar_imm(dtype, saturate, has_mask):
    if saturate and dtype.startswith(("f", "bf")):
        pytest.skip("Unsupported OpenCL saturate vadd for float precision")

    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(1, dtype, return_python_type=True)
    mask = rand(n, "bool") if has_mask else [True] * n
    gt_out = get_vadd_gt(dtype, a, b, mask, saturate)

    prim_func = gen_vadd_gentype_scalar_imm(vdtype, mask, saturate, b)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype=dtype)
    prim_func(a, py_out)
    assert_allclose(py_out[mask], gt_out[mask])

    npu_out = np.empty(n, dtype=dtype)
    ex(a, npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


def gen_vadd_gentype_scalar_var(vdtype, var_dtype, mask, saturate, imm_b):
    @S.prim_func
    def vadd_gentype_scalar_var(a: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        B = S.cast(imm_b, var_dtype)
        vc = S.vadd(a[0], B, mask, saturate=saturate)
        c[0] = vc + S.vadd(B, a[0], mask, saturate=saturate)

    return vadd_gentype_scalar_var


@pytest.mark.parametrize("dtype", DTYPE_TUPLE)
@pytest.mark.parametrize("saturate", (True, False))
@pytest.mark.parametrize("has_mask", (True, False))
def test_vadd_gentype_scalar_var(dtype, saturate, has_mask):
    if saturate and dtype.startswith(("f", "bf")):
        pytest.skip("Unsupported OpenCL saturate vadd for float precision")

    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    mask = rand(n, "bool") if has_mask else [True] * n
    dtype2var_dtype = {
        "int8": ("int8",),
        "uint8": ("uint8",),
        "int16": ("int16", "int8", "uint8"),
        "uint16": ("uint16", "uint8"),
        "int32": ("int32", "int16", "uint16"),
        "uint32": ("uint32", "uint16"),
        "float16": ("float16",),
        "float32": ("float32", "float16"),
        "bfloat16": ("bfloat16", "float16"),
    }
    for var_dtype in dtype2var_dtype[dtype]:
        b = rand(1, var_dtype, return_python_type=True)
        gt_out = get_vadd_gt(dtype, a, b, mask, saturate)

        prim_func = gen_vadd_gentype_scalar_var(vdtype, var_dtype, mask, saturate, b)
        bm = BuildManager()
        ex = bm.build(prim_func)

        py_out = np.empty(n, dtype=dtype)
        prim_func(a, py_out)
        assert_allclose(py_out[mask], gt_out[mask])

        npu_out = np.empty(n, dtype=dtype)
        ex(a, npu_out)
        assert_allclose(npu_out[mask], gt_out[mask])


def gen_vadd_diff_sign(vdtype, out_sign, saturate):
    u_vdtype = vdtype.with_uint()
    out_vdtype = vdtype.with_uint() if out_sign == "u" else vdtype

    @S.prim_func
    def vadd_diff_sign(a: S.ptr(vdtype, "global"), b: S.ptr(u_vdtype, "global"), c: S.ptr(out_vdtype, "global")):
        vc = S.vadd(a[0], b[0], out_sign=out_sign, saturate=saturate)
        c[0] = vc + S.vadd(b[0], a[0], out_sign=out_sign, saturate=saturate)

    return vadd_diff_sign


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32"))
@pytest.mark.parametrize("out_sign", ("u", "s"))
@pytest.mark.parametrize("saturate", (True, False))
def test_vadd_diff_sign(dtype, out_sign, saturate):
    out_dtype = f"u{dtype}" if out_sign == "u" else dtype
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, f"u{dtype}")
    mask = [True] * n
    gt_out = get_vadd_gt(dtype, a, b, mask, saturate, out_sign=out_sign)

    prim_func = gen_vadd_diff_sign(vdtype, out_sign, saturate)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype=out_dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=out_dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_vadd_diff_sign_imm(out_sign, saturate):
    out_vdtype = "uint8x32" if out_sign == "u" else "int8x32"

    @S.prim_func
    def vadd_diff_sign_imm(a: S.ptr("int8x32", "global"), c: S.ptr(out_vdtype, "global")):
        vc = S.vadd(a[0], 252, out_sign=out_sign, saturate=saturate)
        c[0] = vc + S.vadd(252, a[0], out_sign=out_sign, saturate=saturate)

    return vadd_diff_sign_imm


@pytest.mark.parametrize("out_sign", ("u", "s"))
@pytest.mark.parametrize("saturate", (True, False))
def test_vadd_diff_sign_imm(out_sign, saturate):
    dtype = "int8"
    out_dtype = "uint8" if out_sign == "u" else "int8"
    n = 32
    a = rand(n, dtype)
    b = 252
    mask = [True] * n
    gt_out = get_vadd_gt(dtype, a, b, mask, saturate, out_sign=out_sign)

    prim_func = gen_vadd_diff_sign_imm(out_sign, saturate)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype=out_dtype)
    prim_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=out_dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


def vadd_fail(fail_type):
    @S.prim_func
    def vadd_fail_diff_dtype(a: S.ptr("u8x32", "global"), b: S.ptr("i16x16", "global"), c: S.ptr("u8x32", "global")):
        c[0] = S.vadd(a[0], b[0])

    @S.prim_func
    def vadd_fail_scalar_imm_out_of_range(a: S.ptr("int8x32", "global"), c: S.ptr("int8x32", "global")):
        c[0] = S.vadd(a[0], 300)

    @S.prim_func
    def vadd_fail_scalar_var_out_of_range(a: S.ptr("int8x32", "global"), B: S.int16, c: S.ptr("int8x32", "global")):
        c[0] = S.vadd(a[0], B)

    if fail_type == "different_dtype":
        return vadd_fail_diff_dtype
    if fail_type == "scalar_imm_out_of_range":
        return vadd_fail_scalar_imm_out_of_range

    assert fail_type == "scalar_var_out_of_range"
    return vadd_fail_scalar_var_out_of_range


@pytest.mark.parametrize("fail_type", ("different_dtype", "scalar_imm_out_of_range", "scalar_var_out_of_range"))
def test_vadd_fail(capfd, fail_type):
    prim_func = vadd_fail(fail_type)
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
    test_vadd_gentype_gentype(dtype="int8", saturate=True, has_mask=True)
    test_vadd_gentype_gentype(dtype="float32", saturate=False, has_mask=True)
    test_vadd_gentype_scalar_imm(dtype="int8", saturate=False, has_mask=False)
    test_vadd_gentype_scalar_imm(dtype="bfloat16", saturate=False, has_mask=False)
    test_vadd_gentype_scalar_var(dtype="float32", saturate=False, has_mask=False)
    test_vadd_gentype_scalar_var(dtype="bfloat16", saturate=False, has_mask=False)
    test_vadd_diff_sign(dtype="int8", out_sign="u", saturate=True)
    test_vadd_diff_sign_imm(out_sign="u", saturate=True)
    test_vadd_fail(None, "different_dtype")
