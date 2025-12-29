# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vdpa(in0_vdtype, in1_vdtype, out_vdtype, mask):
    @S.prim_func
    def vdpa_func(a: S.ptr(in0_vdtype, "global"), b: S.ptr(in1_vdtype, "global"), c: S.ptr(out_vdtype, "global")):
        tid = S.get_local_id()
        acc = S.cast(1, out_vdtype)
        c[tid] = S.vdpa(acc, a[tid], b[tid], mask)

    return vdpa_func


def gt_dpa(a, b, acc, mask):
    if len(mask) < len(a):
        mask = np.tile(mask, len(a) // len(mask))
    out_dtype = acc.dtype
    a = np.array(a, dtype=out_dtype)
    b = np.array(b, dtype=out_dtype)
    for i in range(len(acc)):
        for j in range(2):
            xy_idx = 2 * i + j
            if mask[xy_idx]:
                acc[i] += a[xy_idx] * b[xy_idx]
    return acc


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
def test_vdpa(in0_dtype, in1_dtype, out_dtype):
    in0_vdtype = hw_native_vdtype(in0_dtype)
    in1_vdtype = hw_native_vdtype(in1_dtype)
    out_vdtype = hw_native_vdtype(out_dtype)

    n, out_n = in0_vdtype.lanes * 4, out_vdtype.lanes * 4
    a = np.array(range(n), dtype=in0_dtype)
    b = np.array(range(n), dtype=in1_dtype)
    acc = np.ones(out_n, dtype=out_dtype)
    mask = rand(in0_vdtype.lanes, "bool")
    gt_out = acc.copy()
    gt_out = gt_dpa(a, b, gt_out, mask)

    prim_func = gen_vdpa(in0_vdtype, in1_vdtype, out_vdtype, mask)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = acc.copy()
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = acc.copy()
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    (
        ("int8", "int8", "int32"),
        ("int8", "int16", "int16"),
        ("int32", "int32", "int8"),
    ),
)
def test_fail_invalid_dtype_vdpa(capfd, in0_dtype, in1_dtype, out_dtype):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func(
            a: S.ptr(hw_native_vdtype(in0_dtype), "global"),
            b: S.ptr(hw_native_vdtype(in1_dtype), "global"),
            c: S.ptr(hw_native_vdtype(out_dtype), "global"),
        ):
            c[0] = S.vdpa(c[0], a[0], b[0])

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = "Argument type mismatch"
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def gen_vqdpa(in0_vdtype, in1_vdtype, out_vdtype, mask):
    @S.prim_func
    def vdpa_func(a: S.ptr(in0_vdtype, "global"), b: S.ptr(in1_vdtype, "global"), c: S.ptr(out_vdtype, "global")):
        tid = S.get_local_id()
        acc = S.cast(1, out_vdtype)
        c[tid] = S.vqdpa(acc, a[tid], b[tid], mask)

    return vdpa_func


def gt_qdpa(a, b, acc, mask):
    if len(mask) < len(a):
        mask = np.tile(mask, len(a) // len(mask))
    out_dtype = acc.dtype
    a = np.array(a, dtype=out_dtype)
    b = np.array(b, dtype=out_dtype)
    if len(a) / len(acc) == 4:
        # vqdpa integer
        for i in range(len(acc)):
            for j in range(4):
                xy_idx = 4 * i + j
                if mask[xy_idx]:
                    acc[i] += a[xy_idx] * b[xy_idx]
    else:
        # vqdpa float
        for i in range(len(mask)):
            if mask[i]:
                acc[(i // 4) * 2] += a[i] * b[i]
    return acc


def _test_vqdpa(in0_dtype, in1_dtype, out_dtype):
    in0_vdtype = hw_native_vdtype(in0_dtype)
    in1_vdtype = hw_native_vdtype(in1_dtype)
    out_vdtype = hw_native_vdtype(out_dtype)

    n, out_n = in0_vdtype.lanes * 4, out_vdtype.lanes * 4
    a = np.array(range(n), dtype=in0_dtype)
    b = np.array(range(n), dtype=in1_dtype)
    acc = np.ones(out_n, dtype=out_dtype)
    mask = rand(in0_vdtype.lanes, "bool")
    gt_out = acc.copy()
    gt_out = gt_qdpa(a, b, gt_out, mask)

    prim_func = gen_vqdpa(in0_vdtype, in1_vdtype, out_vdtype, mask)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = acc.copy()
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = acc.copy()
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    (
        ("int8", "int8", "int32"),
        ("int8", "uint8", "int32"),
        ("uint8", "int8", "int32"),
        ("uint8", "uint8", "uint32"),
    ),
)
def test_integer_vqdpa(in0_dtype, in1_dtype, out_dtype):
    _test_vqdpa(in0_dtype, in1_dtype, out_dtype)


@pytest.mark.NOT_X3P
@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    (
        ("float16", "float16", "float32"),
        ("bfloat16", "bfloat16", "float32"),
    ),
)
def test_floating_vqdpa(in0_dtype, in1_dtype, out_dtype):
    _test_vqdpa(in0_dtype, in1_dtype, out_dtype)


def _test_fail_invalid_dtype_vqdpa(capfd, in0_dtype, in1_dtype, out_dtype):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func(
            a: S.ptr(hw_native_vdtype(in0_dtype), "global"),
            b: S.ptr(hw_native_vdtype(in1_dtype), "global"),
            c: S.ptr(hw_native_vdtype(out_dtype), "global"),
        ):
            c[0] = S.vqdpa(c[0], a[0], b[0])

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = "Argument type mismatch"
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    (
        ("int8", "int8", "int16"),
        ("int8", "int16", "int32"),
    ),
)
def test_fail_invalid_dtype_integer_vqdpa(capfd, in0_dtype, in1_dtype, out_dtype):
    _test_fail_invalid_dtype_vqdpa(capfd, in0_dtype, in1_dtype, out_dtype)


@pytest.mark.NOT_X3P
def test_fail_invalid_dtype_floating_vqdpa(capfd):
    _test_fail_invalid_dtype_vqdpa(capfd, in0_dtype="float32", in1_dtype="float32", out_dtype="float16")


if __name__ == "__main__":
    test_vdpa(in0_dtype="int8", in1_dtype="int8", out_dtype="int16")
    test_integer_vqdpa(in0_dtype="int8", in1_dtype="int8", out_dtype="int32")
    test_floating_vqdpa(in0_dtype="float16", in1_dtype="float16", out_dtype="float32")
    test_floating_vqdpa(in0_dtype="bfloat16", in1_dtype="bfloat16", out_dtype="float32")
    test_fail_invalid_dtype_vdpa(None, "int8", "int8", "int32")
    test_fail_invalid_dtype_floating_vqdpa(None)
