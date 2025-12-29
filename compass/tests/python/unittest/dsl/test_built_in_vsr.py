# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import re
import pytest
import numpy as np
from tvm import get_range
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vsr_func(vdtype, mask, with_round):
    @S.prim_func
    def vsr_func(x: S.ptr(vdtype, "global"), shift: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vsr(x[0], shift[0], mask[: vdtype.lanes], with_round)
        out[1] = S.vsr(x[1], 1, mask[vdtype.lanes :], with_round)

    return vsr_func


def _vsr(x, shift, mask, with_round):
    if with_round:
        ret = np.around(x * (0.5 ** shift.astype("uint32")))
    else:
        ret = x >> shift
    return np.where(mask, ret, 0).astype(x.dtype)


def get_gt_out(x, shift, mask, with_round, vdtype):
    lanes = vdtype.lanes
    out0 = _vsr(x[:lanes], shift[:lanes], mask[:lanes], with_round)
    out1 = _vsr(x[lanes:], np.array(1), mask[lanes:], with_round)
    return np.concatenate([out0, out1])


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
@pytest.mark.parametrize("with_round", (True, False))
def test_vsr(dtype, with_round):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes * 2

    x = rand(n, dtype)
    min_val, max_val = get_range(dtype)
    x[:5] = [min_val + 1, max_val - 1, 0, max_val, min_val]

    shift = rand(n, dtype)
    shift[:7] = [3, -3, 0, 1, -1, vdtype.bits, vdtype.bits + 3]
    np.random.shuffle(shift)
    mask = rand(n, "bool")

    name = f"vsr_{vdtype}_{'with_round' if with_round else 'no_round'}"
    gt_out = get_gt_out(x, shift, mask, with_round, vdtype)

    prim_func = gen_vsr_func(vdtype, mask, with_round)
    bm = BuildManager()
    ex = bm.build(prim_func, name=name)

    py_out = np.empty(n, dtype)
    prim_func(x, shift, py_out)
    assert_allclose(py_out[mask], gt_out[mask])

    npu_out = np.empty(n, dtype)
    ex(x, shift, npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


def gen_vsr_mask_opt_func(vdtype, mask, with_round):
    @S.prim_func
    def vsr_func(x: S.ptr(vdtype, "global"), shift: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vsr(x[0], shift[0], mask, with_round)

    return vsr_func


@pytest.mark.parametrize("with_round", (True, False))
def test_vsr_mask_opt(with_round):
    dtype = "int8"
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes

    x = rand(n, dtype)
    y = rand(n, dtype, low=0, high=vdtype.bits)
    mask = [True] * n
    gt_out = _vsr(x, y, mask, with_round)

    prim_func = gen_vsr_mask_opt_func(vdtype, mask, with_round)
    bm = BuildManager()
    ex = bm.build(prim_func)

    expects = (r"__vasr\(x\[0\], shift\[0\]\)",)  # , r"__vasr\(out0, 1\)")
    if with_round:
        expects = (r"__vasrr\(x\[0\], shift\[0\]\)",)  # , r"__vasrr\(out0, 1\)")
    c_code = ex.c_code
    for expect in expects:
        matches = re.search(expect, c_code, re.MULTILINE)
        assert matches is not None, f"\nExpect snippet:\n{expect}\n\nCompass C code:\n{c_code}\n"

    npu_out = np.empty(n, dtype)
    ex(x, y, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vsr("int8", with_round=False)
    test_vsr("int32", with_round=True)
    test_vsr_mask_opt(with_round=True)
