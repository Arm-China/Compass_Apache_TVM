# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import get_range
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vdot(in0_vdtype, in1_vdtype, out_vdtype, mask):
    @S.prim_func
    def vdot_func(a: S.ptr(in0_vdtype, "global"), b: S.ptr(in1_vdtype, "global"), c: S.ptr(out_vdtype, "global")):
        c[0] = S.vdot(a[0], b[0], mask)

    return vdot_func


def gt_dot(a, b, mask, ret_vdtype):
    a = a.astype("int64")
    b = b.astype("int64")
    out = [0] * ret_vdtype.lanes
    for i in range(len(mask)):
        if mask[i]:
            out[i // 2] += a[i] * b[i]
    return np.clip(out, *get_range(ret_vdtype.element_of)).astype(ret_vdtype.element_of)


@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    (
        ("int8", "int8", "int16"),
        ("int8", "uint8", "int16"),
        ("uint8", "int8", "int16"),
        ("uint8", "uint8", "uint16"),
        ("int16", "int16", "int32"),
        ("int16", "uint16", "int32"),
        ("uint16", "int16", "int32"),
        ("uint16", "uint16", "uint32"),
    ),
)
def test_all_vdot(in0_dtype, in1_dtype, out_dtype):
    in0_vdtype = hw_native_vdtype(in0_dtype)
    in1_vdtype = hw_native_vdtype(in1_dtype)
    out_vdtype = hw_native_vdtype(out_dtype)

    n, out_n = in0_vdtype.lanes, out_vdtype.lanes
    a = rand(n, in0_dtype)
    b = rand(n, in1_dtype)
    mask = rand(n, "bool")
    gt_out = gt_dot(a, b, mask, out_vdtype)
    mask_out = mask[::2] | mask[1::2]

    prim_func = gen_vdot(in0_vdtype, in1_vdtype, out_vdtype, mask)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(out_n, out_dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out[mask_out], gt_out[mask_out])

    npu_out = np.empty(out_n, out_dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out[mask_out], gt_out[mask_out])


@pytest.mark.parametrize(
    "in0_dtype, in1_dtype",
    (
        ("int8", "int16"),
        ("int32", "int32"),
        ("float16", "float16"),
    ),
)
def test_fail_invalid_dtype_vdot(capfd, in0_dtype, in1_dtype):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func(
            a: S.ptr(hw_native_vdtype(in0_dtype), "global"),
            b: S.ptr(hw_native_vdtype(in1_dtype), "global"),
            c: S.ptr(hw_native_vdtype("int32"), "global"),
        ):
            c[0] = S.vdot(a[0], b[0])

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    if in0_dtype == "int8" and in1_dtype == "int16":
        expect = "Argument type mismatch"
    else:
        expect = "Only supports 8-bit or 16-bit integer instruction"
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def gen_vqdot(in0_vdtype, in1_vdtype, out_vdtype, mask):
    @S.prim_func
    def vqdot_func(a: S.ptr(in0_vdtype, "global"), b: S.ptr(in1_vdtype, "global"), c: S.ptr(out_vdtype, "global")):
        c[0] = S.vqdot(a[0], b[0], mask)

    return vqdot_func


def gt_qdot(a, b, mask, ret_vdtype):
    a = a.astype("int64" if ret_vdtype.is_integer else "float64")
    b = b.astype("int64" if ret_vdtype.is_integer else "float64")
    out = np.zeros(ret_vdtype.lanes)
    if ret_vdtype.is_float:
        # vqdot float
        for i in range(len(mask)):
            if mask[i]:
                out[(i // 4) * 2] += a[i] * b[i]
    else:
        # vqdot integer
        for i in range(ret_vdtype.lanes):
            for j in range(4):
                xy_idx = 4 * i + j
                if mask[xy_idx]:
                    out[i] += a[xy_idx] * b[xy_idx]
        out = np.clip(out, *get_range(ret_vdtype.element_of))
    return out.astype(ret_vdtype.element_of)


@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    (
        ("int8", "int8", "int32"),
        ("int8", "uint8", "int32"),
        ("uint8", "int8", "int32"),
        ("uint8", "uint8", "uint32"),
        ("float16", "float16", "float32"),
        ("bfloat16", "bfloat16", "float32"),
    ),
)
def test_all_vqdot(in0_dtype, in1_dtype, out_dtype):
    in0_vdtype = hw_native_vdtype(in0_dtype)
    in1_vdtype = hw_native_vdtype(in1_dtype)
    out_vdtype = hw_native_vdtype(out_dtype)

    n, out_n = in0_vdtype.lanes, out_vdtype.lanes
    a = rand(n, in0_dtype)
    b = rand(n, in1_dtype)
    mask = rand(n, "bool")
    gt_out = gt_qdot(a, b, mask, out_vdtype)
    mask_out = mask[::4] | mask[1::4] | mask[2::4] | mask[3::4]
    mask_out = np.ravel(list(zip(mask_out, [False] * 4))) if out_dtype == "float32" else mask_out

    prim_func = gen_vqdot(in0_vdtype, in1_vdtype, out_vdtype, mask)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(out_n, out_dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out[mask_out], gt_out[mask_out])

    npu_out = np.empty(out_n, out_dtype)
    ex(a, b, npu_out)

    # For BF16, detailed see CP-23944.
    rtol = 1e-3 if in0_dtype == "bfloat16" else None
    atol = 1 if in0_dtype == "float16" else None
    assert_allclose(npu_out[mask_out], gt_out[mask_out], rtol=rtol, atol=atol)


def test_vqdot_bf16_corner_case():
    in0_dtype, in1_dtype, out_dtype = "bfloat16", "bfloat16", "float32"
    in0_vdtype = hw_native_vdtype(in0_dtype)
    in1_vdtype = hw_native_vdtype(in1_dtype)
    out_vdtype = hw_native_vdtype(out_dtype)

    n, out_n = in0_vdtype.lanes, out_vdtype.lanes
    a = np.array([0xC99D, 0x49AB, 0xCA29, 0xC90E] + [0] * 12, "uint16").view(in0_dtype)
    b = np.array([0x4865, 0xC993, 0xC930, 0x4304] + [0] * 12, "uint16").view(in1_dtype)
    mask = np.array([True] * n, bool)
    gt_out = gt_qdot(a, b, mask, out_vdtype)

    prim_func = gen_vqdot(in0_vdtype, in1_vdtype, out_vdtype, mask)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(out_n, out_dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out[0], gt_out[0])

    npu_out = np.empty(out_n, out_dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out[0], gt_out[0], rtol=1e-3)

    assert gt_out[0].view("uint32") == 1340044864
    assert py_out[0].view("uint32") == 1340044864
    assert npu_out[0].view("uint32") == 1340045056


@pytest.mark.parametrize(
    "in0_dtype, in1_dtype",
    (
        ("int8", "int16"),
        ("int16", "int16"),
        ("float32", "float32"),
    ),
)
def test_fail_invalid_dtype_vqdot(capfd, in0_dtype, in1_dtype):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func(
            a: S.ptr(hw_native_vdtype(in0_dtype), "global"),
            b: S.ptr(hw_native_vdtype(in1_dtype), "global"),
            c: S.ptr(hw_native_vdtype("int32"), "global"),
        ):
            c[0] = S.vqdot(a[0], b[0])

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    if in0_dtype == "int8" and in1_dtype == "int16":
        expect = "Argument type mismatch"
    else:
        expect = "Only supports 8-bit integer instruction or 16-bit floating instruction"
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


if __name__ == "__main__":
    test_all_vdot(in0_dtype="int8", in1_dtype="int8", out_dtype="int16")
    test_fail_invalid_dtype_vdot(None, "int8", "int16")
    test_all_vqdot(in0_dtype="int8", in1_dtype="int8", out_dtype="int32")
    test_all_vqdot(in0_dtype="float16", in1_dtype="float16", out_dtype="float32")
    test_all_vqdot(in0_dtype="bfloat16", in1_dtype="bfloat16", out_dtype="float32")
    test_fail_invalid_dtype_vqdot(None, "int8", "int16")
    test_vqdot_bf16_corner_case()
