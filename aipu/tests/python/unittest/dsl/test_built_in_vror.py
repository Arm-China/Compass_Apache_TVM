# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import re
import pytest
import numpy as np
from tvm import aipu, get_range
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_vror_func(vdtype, mask):
    @S.prim_func
    def vror_func(x: S.ptr(vdtype, "global"), shift: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vror(x[0], shift[0], mask[: vdtype.lanes])
        out[1] = S.vror(x[1], 1, mask[vdtype.lanes :])

    return vror_func


def _vror(x, shift, mask, vdtype):
    if vdtype.is_int:
        x = x.astype(vdtype.with_uint().element_of)
    shift = shift % vdtype.bits
    ret = (x >> shift) | (x << (vdtype.bits - shift))
    return np.where(mask, ret, 0).astype(vdtype.element_of)


def get_gt_out(x, shift, mask, vdtype):
    lanes = vdtype.lanes
    out0 = _vror(x[:lanes], shift[:lanes], mask[:lanes], vdtype)
    out1 = _vror(x[lanes:], np.array(1), mask[lanes:], vdtype)
    return np.concatenate([out0, out1])


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_vror(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes * 2

    x = rand(n, dtype)
    min_val, max_val = get_range(dtype)
    x[:5] = [min_val + 1, max_val - 1, 0, max_val, min_val]

    shift = rand(n, dtype)
    shift[:7] = [3, -3, 0, 1, -1, vdtype.bits, vdtype.bits + 3]
    np.random.shuffle(shift)
    mask = rand(n, "bool")
    gt_out = get_gt_out(x, shift, mask, vdtype)

    prim_func = gen_vror_func(vdtype, mask)
    bm = aipu.tir.BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(x, shift, py_out)
    testing.assert_allclose(py_out[mask], gt_out[mask])

    aipu_out = np.empty(n, dtype)
    ex(x, shift, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


def test_vror_mask_opt():
    dtype = "uint8"
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes

    x = rand(n, dtype)
    shift = rand(n, dtype, low=0, high=vdtype.bits)
    mask = [True] * n
    gt_out = get_gt_out(x, shift, mask, vdtype)

    prim_func = gen_vror_func(vdtype, mask)
    bm = aipu.tir.BuildManager()
    ex = bm.build(prim_func)

    expect = r"__vror\(x\[0\], shift\[0\]\)"
    matches = re.search(expect, ex.c_code, re.MULTILINE)
    assert matches is not None, f"\nExpect snippet:\n{expect}\n\nAIPU C code:\n{ex.c_code}\n"

    aipu_out = np.empty(n, dtype)
    ex(x, shift, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


if __name__ == "__main__":
    test_vror("int8")
    test_vror_mask_opt()
