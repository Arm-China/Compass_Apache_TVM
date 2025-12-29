# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import re
import pytest
import numpy as np
from tvm import get_range, DataType
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_combine_vnsr(from_vdtype, to_vdtype, saturate, with_round):
    @S.prim_func
    def combine_vnsr(
        a: S.ptr(from_vdtype, "global"), out: S.ptr(to_vdtype, "global"), shift: S.ptr(from_vdtype, "global")
    ):
        out[0] = S.cast(S.vsr(a[0], shift[0], with_round=with_round), to_vdtype, saturate=saturate)

    return combine_vnsr


def get_combine_vnsr_gt_out(inp, shift, out_dtype, saturate, with_round):
    if with_round:
        ret = np.around(inp * (0.5 ** shift.astype("uint32"))).astype("int64")
    else:
        ret = inp >> shift
    if saturate:
        ret = np.clip(ret, *get_range(out_dtype))
    return ret.astype(out_dtype)


@pytest.mark.parametrize("with_round", (True, False))
@pytest.mark.parametrize("saturate", (True, False))
@pytest.mark.parametrize("to_dtype", ("int8", "uint8", "int16", "uint16"))
@pytest.mark.parametrize("from_dtype", ("int16", "uint16", "int32", "uint32"))
def test_opt_combine_vnsr(from_dtype, to_dtype, saturate, with_round):
    from_vdtype, to_vdtype = hw_native_vdtype(from_dtype), hw_native_vdtype(to_dtype)
    if from_vdtype.bits <= to_vdtype.bits:
        pytest.skip("Invalid combination.")

    from_n = from_vdtype.lanes
    to_n = to_vdtype.lanes
    a = rand(from_n, from_dtype)
    shift = rand(from_n, from_dtype, low=4, high=(from_vdtype.bits - 2))
    gt_out = np.resize(get_combine_vnsr_gt_out(a, shift, to_dtype, saturate, with_round), to_n)

    py_func = gen_combine_vnsr(from_vdtype, to_vdtype, saturate, with_round)
    bm = BuildManager()
    ex = bm.build(py_func)

    expect = "vnsr"
    unexpect = "vasr" if from_vdtype.is_int else "vlsr"
    c_code = ex.c_code
    msg = f"\nExpect snippet:\n{expect}\nUnexpect snippet:\n{unexpect}\n\nCompass C code:\n{c_code}\n"
    assert expect in c_code and unexpect not in c_code, msg

    npu_out = np.empty(to_n, dtype=to_dtype)
    ex(a, npu_out, shift)
    assert_allclose(npu_out[:from_n], gt_out[:from_n])


def gen_combine_vnsr_with_merge(from_vdtype, to_vdtype, saturate, with_round):
    @S.prim_func
    def combine_vnsr_with_merge2(
        a: S.ptr(from_vdtype, "global"), out: S.ptr(to_vdtype, "global"), shift: S.ptr(from_vdtype, "global")
    ):
        out[0] = S.cast(
            (S.vsr(a[0], shift[0], with_round=with_round), S.vsr(a[1], shift[1], with_round=with_round)),
            to_vdtype,
            saturate=saturate,
        )
        out[1] = S.cast(
            (S.vsr(a[2], shift[2], with_round=with_round), S.vsr(a[3], shift[3], with_round=with_round)),
            to_vdtype,
            saturate=saturate,
        )
        out[2] = S.cast(
            (S.vsr(a[4], shift[4], with_round=with_round), S.vsr(a[5], shift[5], with_round=with_round)),
            to_vdtype,
            saturate=saturate,
        )

    @S.prim_func
    def combine_vnsr_with_merge4(
        a: S.ptr(from_vdtype, "global"), out: S.ptr(to_vdtype, "global"), shift: S.ptr(from_vdtype, "global")
    ):
        out[0] = S.cast(
            (
                S.vsr(a[0], shift[0], with_round=with_round),
                S.vsr(a[1], shift[1], with_round=with_round),
                S.vsr(a[2], shift[2], with_round=with_round),
                S.vsr(a[3], shift[3], with_round=with_round),
            ),
            to_vdtype,
            saturate=saturate,
        )

        cur_out = (out + 1).as_ptr(to_vdtype.element_of)
        va0 = S.cast(
            (
                S.vsr(a[4], shift[4], with_round=with_round),
                S.vsr(a[5], shift[5], with_round=with_round),
                S.vsr(a[6], shift[6], with_round=with_round),
            ),
            to_vdtype,
            saturate=saturate,
        )
        S.vstore(va0, cur_out, mask="24T8F")

        cur_out = cur_out + 24
        va1 = S.cast(
            (
                S.vsr(a[7], shift[7], with_round=with_round),
                S.vsr(a[8], shift[8], with_round=with_round),
                S.vsr(a[9], shift[9], with_round=with_round),
            ),
            to_vdtype,
            saturate=saturate,
        )
        S.vstore(va1, cur_out, mask="24T8F")

        cur_out = cur_out + 24
        va2 = S.cast(
            (S.vsr(a[10], shift[10], with_round=with_round), S.vsr(a[11], shift[11], with_round=with_round)),
            to_vdtype,
            saturate=saturate,
        )
        S.vstore(va2, cur_out, mask="16T16F")

    if from_vdtype.bits // to_vdtype.bits == 2:
        return combine_vnsr_with_merge2
    return combine_vnsr_with_merge4


@pytest.mark.parametrize("with_round", (True, False))
@pytest.mark.parametrize("saturate", (True, False))
@pytest.mark.parametrize("to_dtype", ("int8", "uint8", "int16", "uint16"))
@pytest.mark.parametrize("from_dtype", ("int16", "uint16", "int32", "uint32"))
def test_opt_combine_vnsr_with_merge(from_dtype, to_dtype, saturate, with_round):
    from_vdtype, to_vdtype = hw_native_vdtype(from_dtype), hw_native_vdtype(to_dtype)
    if from_vdtype.bits <= to_vdtype.bits:
        pytest.skip("Invalid combination.")

    n = to_vdtype.lanes * 3
    a = rand(n, from_dtype)
    shift = rand(n, from_dtype, low=4, high=(from_vdtype.bits - 2))
    gt_out = get_combine_vnsr_gt_out(a, shift, to_dtype, saturate, with_round)

    py_func = gen_combine_vnsr_with_merge(from_vdtype, to_vdtype, saturate, with_round)
    bm = BuildManager()
    ex = bm.build(py_func)

    expect = "vnsr"
    unexpect = "vasr" if from_vdtype.is_int else "vlsr"
    c_code = ex.c_code
    msg = f"\nExpect snippet:\n{expect}\nUnexpect snippet:\n{unexpect}\n\nCompass C code:\n{c_code}\n"
    assert expect in c_code and unexpect not in c_code, msg

    npu_out = np.empty(n, dtype=to_dtype)
    ex(a, npu_out, shift)
    assert_allclose(npu_out, gt_out)


def gen_combine_vmull_vmulh(a_dtype, b_dtype, out_dtype):
    lanes0 = 11
    lanes1 = 29
    lanes2 = 32
    lanes3 = 35

    @S.prim_func
    def combine_vmull_vmulh(
        inp0: S.ptr(a_dtype, "global"), inp1: S.ptr(b_dtype, "global"), out: S.ptr(out_dtype, "global")
    ):
        if S.get_local_id() != 0:
            return

        cur_inp0, cur_inp1, cur_out = inp0, inp1, out
        # 1. lanes=11
        va0 = S.vload(cur_inp0, lanes=lanes0)
        vb0 = S.vload(cur_inp1, lanes=lanes0)
        vo0 = S.cast(va0, out_dtype) * S.cast(vb0, out_dtype)
        S.vstore(vo0, cur_out)

        cur_inp0, cur_inp1, cur_out = cur_inp0 + lanes0, cur_inp1 + lanes0, cur_out + lanes0
        # 2. lanes=29
        va1 = S.vload(cur_inp0, lanes=lanes1)
        vb1 = S.vload(cur_inp1, lanes=lanes1)
        vo1 = S.vmul(S.cast(va1, out_dtype), S.cast(vb1, out_dtype), S.tail_mask(8, lanes1))
        S.vstore(vo1, cur_out)

        cur_inp0, cur_inp1, cur_out = cur_inp0 + lanes1, cur_inp1 + lanes1, cur_out + lanes1
        # 3. lanes=32
        va2 = S.vload(cur_inp0, lanes=lanes2)
        vb2 = S.vload(cur_inp1, lanes=lanes2)
        mask2 = S.cast(va2, out_dtype) > S.cast(vb2, out_dtype)
        vo2 = S.vmul(S.cast(va2, out_dtype), S.cast(vb2, out_dtype), mask2)
        S.vstore(vo2, cur_out)

        cur_inp0, cur_inp1, cur_out = cur_inp0 + lanes2, cur_inp1 + lanes2, cur_out + lanes2
        # 3. lanes=35
        va3 = S.vload(cur_inp0, lanes=lanes3)
        vb3 = S.vload(cur_inp1, lanes=lanes3)
        mask3 = S.cast(va3, out_dtype) > S.cast(vb3, out_dtype)
        vo3 = S.vmul(S.cast(va3, out_dtype), S.cast(vb3, out_dtype), mask3, r=S.cast(va3, out_dtype))
        S.vstore(vo3, cur_out)

    return combine_vmull_vmulh


def cast_helper(from_dtype, to_dtype, x):
    if from_dtype.is_float and to_dtype.is_integer:
        x = np.where(np.isnan(x), 0, x)
        # Here will promote to "float64" automatically, so it's safe.
        x = np.clip(x, *get_range("int32"))
    return x.astype(to_dtype.element_of)


def get_gt_out_vmull_vmulh(a, b, out_dtype):
    a_dtype, b_dtype = DataType(a.dtype), DataType(b.dtype)
    a = cast_helper(a_dtype, out_dtype, a)
    b = cast_helper(b_dtype, out_dtype, b)

    # Deal with b is scalar.
    b = np.broadcast_to(b, a.shape)
    ret = a * b
    ret[72:] = np.where(a[72:] > b[72:], ret[72:], a[72:])
    return ret


@pytest.mark.parametrize(
    "a_dtype, b_dtype, out_dtype",
    (
        ("int8", "int8", "int16"),
        ("int8", "int8", "uint16"),
        ("int8", "uint8", "int16"),
        ("int8", "int8", "int32"),
        ("uint8", "int8", "uint32"),
        ("int16", "int16", "int32"),
        ("int16", "int16", "uint32"),
        ("uint16", "int16", "uint32"),
        # Below are cases that can't trigger this optimization.
        ("int8", "int16", "uint16"),
        ("uint16", "int8", "int32"),
        ("uint16", "int16", "uint16"),
        ("int32", "uint32", "uint16"),
        ("int32", "int8", "int16"),
        ("int16", "float16", "int32"),
        ("float16", "int32", "float32"),
        ("float16", "float16", "float32"),
    ),
)
def test_opt_combine_vmull_vmulh(a_dtype, b_dtype, out_dtype):
    a_dtype, b_dtype, out_dtype = DataType(a_dtype), DataType(b_dtype), DataType(out_dtype)
    n = 11 + 29 + 32 + 35
    a = rand(n, a_dtype)
    b = rand(n, b_dtype)
    mask_head = np.array([True] * 11 + ([True] * 8 + [False] * 21))
    mask_body = cast_helper(a_dtype, out_dtype, a)[40:72] > cast_helper(b_dtype, out_dtype, b)[40:72]
    mask_tail = np.array([True] * 35)
    mask = np.hstack((mask_head, mask_body, mask_tail))
    gt_out = get_gt_out_vmull_vmulh(a, b, out_dtype)

    py_func = gen_combine_vmull_vmulh(a_dtype, b_dtype, out_dtype)
    ex = BuildManager().build(py_func)

    expects = (
        r"= ((?!(__vexte|__vmul)).)*__vmulh",
        r"= __vsel((?!(__vexte|__vmul)).)*__vmulh",
    )

    for expect in expects:
        matches = re.search(expect, ex.c_code, re.MULTILINE)
        if (
            (a_dtype.is_integer and b_dtype.is_integer and out_dtype.is_integer)
            and a_dtype.bits == b_dtype.bits
            and a_dtype.bits < out_dtype.bits
        ):
            assert matches is not None, f"\nExpect snippet:\n{expect}\n\nCompass C code:\n{ex.c_code}\n"
        else:
            assert matches is None, f"\nUnexpect snippet:\n{expect}\n\nCompass C code:\n{ex.c_code}\n"

    npu_out = np.empty(n, dtype=str(out_dtype))
    ex(a, b, npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


def gen_combine_vmull_vmulh_vbcast(a_dtype, b_dtype, out_dtype):
    lanes0 = 11
    lanes1 = 29
    lanes2 = 32
    lanes3 = 35

    @S.prim_func
    def combine_vmull_vmulh_vbcast(
        inp0: S.ptr(a_dtype, "global"), inp1: S.ptr(b_dtype, "global"), out: S.ptr(out_dtype, "global")
    ):
        if S.get_local_id() != 0:
            return

        scalar_b = inp1[0]

        cur_inp0, cur_out = inp0, out
        # 1. lanes=11
        va0 = S.vload(cur_inp0, lanes=lanes0)
        vo0 = S.cast(va0, out_dtype) * S.cast(scalar_b, out_dtype)
        S.vstore(vo0, cur_out)

        cur_inp0, cur_out = cur_inp0 + lanes0, cur_out + lanes0
        # 2. lanes=29
        va1 = S.vload(cur_inp0, lanes=lanes1)
        vo1 = S.vmul(S.cast(va1, out_dtype), S.cast(scalar_b, out_dtype), S.tail_mask(8, lanes1))
        S.vstore(vo1, cur_out)

        cur_inp0, cur_out = cur_inp0 + lanes1, cur_out + lanes1
        # 3. lanes=32
        va2 = S.vload(cur_inp0, lanes=lanes2)
        mask2 = S.cast(va2, out_dtype) > S.cast(scalar_b, out_dtype)
        vo2 = S.vmul(S.cast(va2, out_dtype), S.cast(scalar_b, out_dtype), mask2)
        S.vstore(vo2, cur_out)

        cur_inp0, cur_out = cur_inp0 + lanes2, cur_out + lanes2
        # 3. lanes=35
        va3 = S.vload(cur_inp0, lanes=lanes3)
        mask3 = S.cast(va3, out_dtype) > S.cast(scalar_b, out_dtype)
        vo3 = S.vmul(S.cast(va3, out_dtype), S.cast(scalar_b, out_dtype), mask3, r=S.cast(va3, out_dtype))
        S.vstore(vo3, cur_out)

    return combine_vmull_vmulh_vbcast


@pytest.mark.parametrize(
    "a_dtype, b_dtype, out_dtype",
    (
        ("int8", "int8", "int16"),
        ("int8", "int8", "uint16"),
        ("int8", "uint8", "int16"),
        ("int8", "int8", "int32"),
        ("uint8", "int8", "uint32"),
        ("int16", "int16", "int32"),
        ("int16", "int16", "uint32"),
        ("uint16", "int16", "uint32"),
        # Below are cases that can't trigger this optimization.
        ("int8", "int16", "uint16"),
        ("uint16", "int8", "int32"),
        ("uint16", "int16", "uint16"),
        ("int32", "uint32", "uint16"),
        ("int32", "int8", "int16"),
        ("int16", "float16", "int32"),
        ("float16", "int32", "float32"),
        ("float16", "float16", "float32"),
    ),
)
def test_opt_combine_vmull_vmulh_vbcast(a_dtype, b_dtype, out_dtype):
    a_dtype, b_dtype, out_dtype = DataType(a_dtype), DataType(b_dtype), DataType(out_dtype)
    n = 11 + 29 + 32 + 35
    a = rand(n, a_dtype)
    b = np.full(n, rand(1, b_dtype))
    mask_head = np.array([True] * 11 + ([True] * 8 + [False] * 21))
    mask_body = cast_helper(a_dtype, out_dtype, a)[40:72] > cast_helper(b_dtype, out_dtype, b)[40:72]
    mask_tail = np.array([True] * 35)
    mask = np.hstack((mask_head, mask_body, mask_tail))
    gt_out = get_gt_out_vmull_vmulh(a, b[0], out_dtype)

    py_func = gen_combine_vmull_vmulh_vbcast(a_dtype, b_dtype, out_dtype)
    ex = BuildManager(disabled_pass=["tir.CommonSubexprElimTIR"]).build(py_func)

    expects = (
        r"= ((?!(__vexte|__vmul)).)*__vmulh",
        r"= __vsel((?!(__vexte|__vmul)).)*__vmulh",
    )

    for expect in expects:
        matches = re.search(expect, ex.c_code, re.MULTILINE)
        if (
            (a_dtype.is_integer and b_dtype.is_integer and out_dtype.is_integer)
            and a_dtype.bits == b_dtype.bits
            and a_dtype.bits < out_dtype.bits
        ):
            assert matches is not None, f"\nExpect snippet:\n{expect}\n\nCompass C code:\n{ex.c_code}\n"
        else:
            assert matches is None, f"\nUnexpect snippet:\n{expect}\n\nCompass C code:\n{ex.c_code}\n"

    npu_out = np.empty(n, dtype=str(out_dtype))
    ex(a, b, npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


def gen_combine_vbclr(dtype, mask, is_ab_mask):
    @S.prim_func
    def combine_vbclr_func(inp0: S.ptr(dtype, "global"), inp1: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        va = S.vload(inp0)
        vb = S.vload(inp1)
        if is_ab_mask:
            mask_a = va > 0
            mask_b = vb > 0
            mask_out = S.vand(mask_a, S.vinv(mask_b), mask)
            vout = S.vsel(va, vb, mask_out)
            S.vstore(vout, out)
        else:
            vout = S.vand(va, S.vinv(vb), mask)
            S.vstore(vout, out)

    return combine_vbclr_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
@pytest.mark.parametrize("is_all_true_mask", (True, False))
@pytest.mark.parametrize("is_ab_mask", (True, False))
def test_combine_vbclr(dtype, is_all_true_mask, is_ab_mask):
    vdtype = hw_native_vdtype(dtype)
    pytest.mark.skipif(vdtype.is_uint and is_ab_mask)
    n = vdtype.lanes
    x = rand(n, dtype)
    y = rand(n, dtype)
    mask = [True] * n if is_all_true_mask else rand(n, "bool")
    if is_ab_mask:
        mask_out = np.where(mask, (x > 0) & (~(y > 0)), False)
        gt_out = np.where(mask_out, x, y)
    else:
        gt_out = np.where(mask, x & (~y), 0)

    py_func = gen_combine_vbclr(dtype, mask, is_ab_mask)
    bm = BuildManager()
    ex = bm.build(py_func)

    expect = "vbclr"
    unexpects = ["vand", "vinv"]
    c_code = ex.c_code
    msg = f"\nExpect snippet:\n{expect}\nUnexpect snippet:\n{unexpects}\n\nCompass C code:\n{c_code}\n"
    assert expect in c_code and all(x not in c_code for x in unexpects), msg

    npu_out = np.empty(n, dtype)
    ex(x, y, npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


def gen_combine_vbset(dtype, mask, is_ab_mask):
    @S.prim_func
    def combine_vbset_func(inp0: S.ptr(dtype, "global"), inp1: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        va = S.vload(inp0)
        vb = S.vload(inp1)
        if is_ab_mask:
            mask_a = va > 0
            mask_b = vb > 0
            mask_out = S.vor(mask_a, S.vinv(mask_b), mask)
            vout = S.vsel(va, vb, mask_out)
            S.vstore(vout, out)
        else:
            vout = S.vor(va, S.vinv(vb), mask)
            S.vstore(vout, out)

    return combine_vbset_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
@pytest.mark.parametrize("is_all_true_mask", (True, False))
@pytest.mark.parametrize("is_ab_mask", (True, False))
def test_combine_vbset(dtype, is_all_true_mask, is_ab_mask):
    vdtype = hw_native_vdtype(dtype)
    pytest.mark.skipif(vdtype.is_uint and is_ab_mask)
    n = vdtype.lanes
    x = rand(n, dtype)
    y = rand(n, dtype)
    mask = [True] * n if is_all_true_mask else rand(n, "bool")
    if is_ab_mask:
        mask_out = np.where(mask, (x > 0) | (~(y > 0)), False)
        gt_out = np.where(mask_out, x, y)
    else:
        gt_out = np.where(mask, x | (~y), 0)

    py_func = gen_combine_vbset(dtype, mask, is_ab_mask)
    bm = BuildManager()
    ex = bm.build(py_func)

    expect = "vbset"
    unexpects = ["vor", "vinv"]
    c_code = ex.c_code
    msg = f"\nExpect snippet:\n{expect}\nUnexpect snippet:\n{unexpects}\n\nCompass C code:\n{c_code}\n"
    assert expect in c_code and all(x not in c_code for x in unexpects), msg

    npu_out = np.empty(n, dtype)
    ex(x, y, npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


def gen_combine_vnand_vor(func_name, vdtype, mask_vand_vor, mask_vinv):
    sdot_vand_or_vor = S.vand if func_name == "vnand" else S.vor

    @S.prim_func
    def combine_func(inp0: S.ptr(vdtype, "global"), inp1: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        va = S.vload(inp0)
        vb = S.vload(inp1)
        mask_a = va > 0
        mask_b = vb > 0
        mask_out = S.vinv(sdot_vand_or_vor(mask_a, mask_b, mask_vand_vor), mask_vinv)
        vout = S.vsel(va, vb, mask_out)
        S.vstore(vout, out)

    return combine_func


def get_vnand_vnor_gt(func_name, x, y, mask_vand_vor, mask_vinv):
    mask_a = x > 0
    mask_b = y > 0
    mask_out = mask_a & mask_b if func_name == "vnand" else mask_a | mask_b
    mask_out0 = np.where(mask_vand_vor, mask_out, False)
    mask_out1 = np.where(mask_vinv, ~mask_out0, False)
    return np.where(mask_out1, x, y)


@pytest.mark.parametrize("func_name", ("vnand", "vnor"))
@pytest.mark.parametrize(
    "is_combine, is_mask_var",
    (
        (True, False),
        (False, True),
        (False, False),
    ),
)
def test_combine_vnand_vnor(func_name, is_combine, is_mask_var):
    dtype = "int32"
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    x = rand(n, dtype)
    y = rand(n, dtype)
    mask = [True] * n
    if is_combine:
        mask_vand_vor = [True] * n
    else:
        mask_vand_vor = x > 0 if is_mask_var else [False] * n

    gt_out = get_vnand_vnor_gt(func_name, x, y, mask_vand_vor, mask)

    py_func = gen_combine_vnand_vor(func_name, vdtype, mask_vand_vor, mask)
    bm = BuildManager()
    ex = bm.build(py_func)

    expects = (func_name,) if is_combine else ("vinv", f"v{func_name[2:]}")
    c_code = ex.c_code
    msg = f"\nExpect snippet:\n{expects}\n\nCompass C code:\n{c_code}\n"
    assert all(x in c_code for x in expects), msg

    npu_out = np.empty(n, dtype)
    ex(x, y, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_combine_rounded_vcast(dtype, rtype):
    @S.prim_func
    def rounded_vcast_func(inp0: S.ptr(dtype, "global"), out: S.ptr(rtype, "global")):
        va = S.vload(inp0)
        y = S.cast(S.rint(va), rtype)
        S.vstore(y, out)

    return rounded_vcast_func


LEGAL_CAST_TEST_CASES = [
    ("float32", "int32"),
    ("float32", "int16"),
    ("float32", "uint16"),
    ("float32", "int8"),
    ("float32", "uint8"),
    ("float16", "int32"),
    ("float16", "uint32"),
    ("bfloat16", "int32"),
]


@pytest.mark.parametrize("dtype, rtype", LEGAL_CAST_TEST_CASES)
def test_combine_rounded_vcast(dtype, rtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    x = rand(n, dtype, low=-10, high=10)
    x[0] = 1.7

    gt_out = np.round(x).astype(rtype)

    func = gen_combine_rounded_vcast(dtype, rtype)
    bm = BuildManager()
    ex = bm.build(func)

    expects = ("__vcvt",)
    if dtype in ("float16", "bfloat16"):
        if bm.cps_info.version in ("X3P", "X3S"):
            expects = ("__vcvtuh_tw", "__vcvtul_tw")
        else:
            expects = ("__vcvtue_tw", "__vcvtuo_tw")
    c_code = ex.c_code
    msg = f"\nExpect snippet:\n{expects}\n\nCompass C code:\n{c_code}\n"
    assert all(x in c_code for x in expects), msg

    npu_out = np.empty(n, rtype)
    ex(x, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_opt_combine_vnsr("uint16", "int8", True, True)
    test_opt_combine_vnsr_with_merge("int32", "int8", True, True)
    test_opt_combine_vmull_vmulh("int8", "int8", "int16")
    test_opt_combine_vmull_vmulh("int8", "uint8", "int16")
    test_opt_combine_vmull_vmulh("uint16", "int16", "uint32")
    test_opt_combine_vmull_vmulh("int8", "uint8", "int32")
    test_opt_combine_vmull_vmulh_vbcast("int16", "float16", "int32")
    test_combine_vbclr("int32", False, True)
    test_combine_vbset("int32", False, False)
    test_combine_vnand_vnor("vnand", True, False)
    test_combine_vnand_vnor("vnor", False, True)
    test_combine_rounded_vcast("float32", "int32")
    test_combine_rounded_vcast("float16", "int32")
