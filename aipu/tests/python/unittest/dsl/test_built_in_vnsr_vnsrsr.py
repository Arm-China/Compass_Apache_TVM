# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu, get_range
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def get_vps(to_h, bits):
    if bits == 32:
        vp_t = "8TF" if to_h else "8TFFF"
        vp_st = "8T8F" if to_h else "8T24F"  # Also can be written as 'vp_st = "8T"'.
    else:
        vp_t = "16TF"
        vp_st = "16T16F"  # Also can be written as 'vp_st = "16T"'.

    return vp_t, vp_st


def gen_vnsr(vdtype, shift_vdtype, mask, out_dtype, saturate, out_sign, with_round, to_h):
    @S.prim_func
    def vnsr_func(x: S.ptr(vdtype, "global"), shift: S.ptr(shift_vdtype, "global"), out: S.ptr(out_dtype, "global")):
        vp_t, vp_st = S.meta_var(get_vps(to_h, vdtype.bits))

        out0 = S.vcompt(S.vnsr(x[0], shift[0], mask, saturate, out_sign, with_round, to_h), vp_t)
        out1 = S.vcompt(S.vnsr(x[0], 1, mask, saturate, out_sign, with_round, to_h), vp_t)

        S.vstore(out0, out, mask=vp_st)
        S.vstore(out1, out + vdtype.lanes, mask=vp_st)

    return vnsr_func


def get_vnsr_gt_out(x, shift, mask, out_dtype, saturate, with_round):
    if with_round:
        out0 = np.where(mask, np.around(x * (0.5 ** shift.astype("uint32"))), 0).astype("int64")
        out1 = np.where(mask, np.around(x * (0.5**1)), 0).astype("int64")
    else:
        out0 = np.where(mask, np.array([x[i] >> int(shift[i]) for i in range(len(x))]), 0)
        out1 = np.where(mask, np.array(x >> 1), 0)

    if saturate:
        out0 = np.clip(out0, *get_range(out_dtype))
        out1 = np.clip(out1, *get_range(out_dtype))

    out0 = out0.astype(out_dtype)
    out1 = out1.astype(out_dtype)
    return np.concatenate((out0, out1)).reshape(-1)


@pytest.mark.parametrize("dtype", ("int16", "uint16", "int32", "uint32"))
@pytest.mark.parametrize("is_shift_signed", (True, False))
@pytest.mark.parametrize("saturate", (True, False))
@pytest.mark.parametrize("out_sign", ("s", "u", None))
@pytest.mark.parametrize("with_round", (True, False))
def test_vnsr(dtype, is_shift_signed, saturate, out_sign, with_round):
    vdtype = hw_native_vdtype(dtype)
    shift_vdtype = vdtype.with_int() if is_shift_signed else vdtype.with_uint()
    n = vdtype.lanes

    x = rand(n, dtype)
    min_val, max_val = get_range(dtype)
    x[:5] = [min_val + 1, max_val - 1, 0, max_val, min_val]

    shift = rand(n, shift_vdtype.element_of)
    shift[:7] = [3, -3, 0, 1, -1, vdtype.bits, vdtype.bits + 3]
    np.random.shuffle(shift)

    mask = rand(n, "bool")
    out_vdtype = vdtype
    if out_sign is not None:
        out_vdtype = vdtype.with_int() if out_sign == "s" else vdtype.with_uint()

    bm = aipu.tir.BuildManager()

    for to_h in (True, False) if vdtype.bits == 32 else (False,):
        out_dtype = out_vdtype.with_bits(16 if to_h else 8).element_of
        gt_out = get_vnsr_gt_out(x, shift, mask, out_dtype, saturate, with_round)

        f_vnsr = gen_vnsr(vdtype, shift_vdtype, mask, out_dtype, saturate, out_sign, with_round, to_h)
        ex = bm.build(f_vnsr)

        py_out = np.empty(n * 2, out_dtype)
        f_vnsr(x, shift, py_out)
        testing.assert_allclose(py_out, gt_out)

        aipu_out = np.empty(n * 2, out_dtype)
        ex(x, shift, aipu_out)
        testing.assert_allclose(aipu_out, gt_out)


def gen_vnsrsr(vdtype, shift_vdtype, mask, out_dtype, out_sign, to_h):
    @S.prim_func
    def vnsrsr_func(x: S.ptr(vdtype, "global"), shift: S.ptr(shift_vdtype, "global"), out: S.ptr(out_dtype, "global")):
        vp_t, vp_st = S.meta_var(get_vps(to_h, vdtype.bits))

        out0 = S.vcompt(S.vnsrsr(x[0], shift[0], mask, out_sign=out_sign, to_h=to_h), vp_t)
        out1 = S.vcompt(S.vnsrsr(x[0], 1, mask, out_sign=out_sign, to_h=to_h), vp_t)

        S.vstore(out0, out, mask=vp_st)
        S.vstore(out1, out + vdtype.lanes, mask=vp_st)

    return vnsrsr_func


@pytest.mark.parametrize("dtype", ("int16", "uint16", "int32", "uint32"))
@pytest.mark.parametrize("is_shift_signed", (True, False))
@pytest.mark.parametrize("out_sign", ("s", "u", None))
def test_vnsrsr(dtype, is_shift_signed, out_sign):
    vdtype = hw_native_vdtype(dtype)
    shift_vdtype = vdtype.with_int() if is_shift_signed else vdtype.with_uint()
    n = vdtype.lanes

    x = rand(n, dtype)
    min_val, max_val = get_range(dtype)
    x[:5] = [min_val + 1, max_val - 1, 0, max_val, min_val]

    shift = rand(n, shift_vdtype.element_of)
    shift[:7] = [3, -3, 0, 1, -1, vdtype.bits, vdtype.bits + 3]
    np.random.shuffle(shift)

    mask = rand(n, "bool")
    out_vdtype = vdtype
    if out_sign is not None:
        out_vdtype = vdtype.with_int() if out_sign == "s" else vdtype.with_uint()

    bm = aipu.tir.BuildManager()

    for to_h in (True, False) if vdtype.bits == 32 else (False,):
        out_dtype = out_vdtype.with_bits(16 if to_h else 8).element_of
        gt_out = get_vnsr_gt_out(x, shift, mask, out_dtype, saturate=True, with_round=True)

        f_vnsrsr = gen_vnsrsr(vdtype, shift_vdtype, mask, out_dtype, out_sign, to_h)
        ex = bm.build(f_vnsrsr)

        py_out = np.empty(n * 2, out_dtype)
        f_vnsrsr(x, shift, py_out)
        testing.assert_allclose(py_out, gt_out)

        aipu_out = np.empty(n * 2, out_dtype)
        ex(x, shift, aipu_out)
        testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_vnsr("uint16", is_shift_signed=True, saturate=True, out_sign="s", with_round=False)
    test_vnsr("int32", is_shift_signed=True, saturate=False, out_sign="u", with_round=True)
    test_vnsrsr("uint16", is_shift_signed=True, out_sign="s")
    test_vnsrsr("int32", is_shift_signed=False, out_sign="u")
