# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def fold_mask_generic(dtype):
    @S.prim_func
    def fold_mask_func(inp0: S.ptr(dtype, "global"), inp1: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        mask = S.const_mask("4F4T")
        out[0] = S.vadd(inp0[0], inp1[0], mask=mask, r=0)

    return fold_mask_func


def test_opt_fold_mask():
    dtype = "int32"
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)

    gt_out = np.zeros(n, dtype=dtype)
    gt_out[4:] = a[4:] + b[4:]

    py_func = fold_mask_generic(vdtype)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    expect = "__vmov_w(0x11110000)"
    assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nAIPU C code:\n{ex.c_code}\n"

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out[4:], gt_out[4:])


def unfold_mask_generic(dtype):
    @S.prim_func
    def unfold_mask_func(
        inp0: S.ptr(dtype, "global"),
        inp1: S.ptr(dtype, "global"),
        out: S.ptr(dtype, "global"),
        flag: S.i32,
    ):
        mask = S.const_mask("4F4T")
        if flag > 1:
            mask = S.const_mask("4T4F")
        out[0] = S.vadd(inp0[0], inp1[0], mask=mask, r=0)

    return unfold_mask_func


def test_opt_unfold_mask():
    dtype = "int32"
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    flag = 3

    gt_out = np.zeros(n, dtype=dtype)
    gt_out[:4] = a[:4] + b[:4]

    py_func = unfold_mask_generic(vdtype)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    expect = ", mask)"
    assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nAIPU C code:\n{ex.c_code}\n"

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, b, aipu_out, flag)
    testing.assert_allclose(aipu_out[:4], gt_out[:4])


if __name__ == "__main__":
    test_opt_fold_mask()
    test_opt_unfold_mask()
