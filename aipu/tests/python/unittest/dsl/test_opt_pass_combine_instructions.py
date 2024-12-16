# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import re
import pytest
import numpy as np
from tvm import aipu, get_range, DataType
from tvm.aipu import script as S, testing
from tvm.aipu.utils import rand, hw_native_vdtype


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
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    expect = "vnsr"
    unexpect = "vasr" if from_vdtype.is_int else "vlsr"
    c_code = ex.c_code
    msg = f"\nExpect snippet:\n{expect}\nUnexpect snippet:\n{unexpect}\n\nAIPU C code:\n{c_code}\n"
    assert expect in c_code and unexpect not in c_code, msg

    aipu_out = np.empty(to_n, dtype=to_dtype)
    ex(a, aipu_out, shift)
    testing.assert_allclose(aipu_out[:from_n], gt_out[:from_n])


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
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    expect = "vnsr"
    unexpect = "vasr" if from_vdtype.is_int else "vlsr"
    c_code = ex.c_code
    msg = f"\nExpect snippet:\n{expect}\nUnexpect snippet:\n{unexpect}\n\nAIPU C code:\n{c_code}\n"
    assert expect in c_code and unexpect not in c_code, msg

    aipu_out = np.empty(n, dtype=to_dtype)
    ex(a, aipu_out, shift)
    testing.assert_allclose(aipu_out, gt_out)


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


def cast_helper(from_dtype_str, to_dtype_str, x):
    from_dtype = DataType(from_dtype_str)
    to_dtype = DataType(to_dtype_str)

    if from_dtype.is_float and to_dtype.is_integer:
        x = np.round(x)
        x = np.where(np.isnan(x), 0, x)
        # Here will promote to "float64" automatically, so it's safe.
        x = np.clip(x, *get_range("int32"))
    return x.astype(to_dtype.element_of)


def get_gt_out_vmull_vmulh(a, b, out_dtype):
    a = cast_helper(a.dtype, out_dtype, a)
    b = cast_helper(b.dtype, out_dtype, b)

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
    n = 11 + 29 + 32 + 35
    a = rand(n, a_dtype)
    b = rand(n, b_dtype)
    mask_head = np.array([True] * 11 + ([True] * 8 + [False] * 21))
    mask_body = cast_helper(a_dtype, out_dtype, a)[40:72] > cast_helper(b_dtype, out_dtype, b)[40:72]
    mask_tail = np.array([True] * 35)
    mask = np.hstack((mask_head, mask_body, mask_tail))
    gt_out = get_gt_out_vmull_vmulh(a, b, out_dtype)

    py_func = gen_combine_vmull_vmulh(a_dtype, b_dtype, out_dtype)
    ex = aipu.tir.BuildManager().build(py_func)

    expects = (
        r"= ((?!(__vexte|__vmul)).)*__vmulh",
        r"= __vsel((?!(__vexte|__vmul)).)*__vmulh",
    )

    a_dtype, b_dtype, out_dtype = DataType(a_dtype), DataType(b_dtype), DataType(out_dtype)
    for expect in expects:
        matches = re.search(expect, ex.c_code, re.MULTILINE)
        if (
            (a_dtype.is_integer and b_dtype.is_integer and out_dtype.is_integer)
            and a_dtype.bits == b_dtype.bits
            and a_dtype.bits < out_dtype.bits
        ):
            assert matches is not None, f"\nExpect snippet:\n{expect}\n\nAIPU C code:\n{ex.c_code}\n"
        else:
            assert matches is None, f"\nUnexpect snippet:\n{expect}\n\nAIPU C code:\n{ex.c_code}\n"

    aipu_out = np.empty(n, dtype=str(out_dtype))
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


if __name__ == "__main__":
    test_opt_combine_vnsr("uint16", "int8", True, True)
    test_opt_combine_vnsr_with_merge("int32", "int8", True, True)
    test_opt_combine_vmull_vmulh("int8", "int8", "int16")
    test_opt_combine_vmull_vmulh("int8", "uint8", "int16")
    test_opt_combine_vmull_vmulh("uint16", "int16", "uint32")
    test_opt_combine_vmull_vmulh("int8", "uint8", "int32")
