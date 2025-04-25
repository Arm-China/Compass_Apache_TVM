# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing


dtype = "int8"


@S.prim_func
def fwv_dtype_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global")):
    va1 = S.i8x26(-2)
    va2 = S.u8x26(3)
    S.vstore(S.vadd(S.vadd(S.vload(a, lanes=26), va1, out_sign="s"), va2, out_sign="s"), b)

    va3 = S.fp32x2(4)
    va4 = S.fp32x2(5)
    S.vstore(S.cast(S.vadd(va3, va4), "i8x2"), b + 26)

    va5 = S.fp16x4(6)
    va6 = S.fp16x4(7)
    S.vstore(S.cast(S.cast(S.vadd(va5, va6), "i32x4"), "i8x4"), b + 28)


def test_fwv_dtype_auto_gen():
    n = 32
    a = np.ones(n, dtype)
    gt_out = np.array([2] * 26 + [9] * 2 + [13] * 4, dtype)

    bm = aipu.tir.BuildManager()
    ex = bm.build(fwv_dtype_func)

    py_out = np.empty(n, dtype)
    fwv_dtype_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def test_fwv_dtype_invalid():
    with pytest.raises(AttributeError) as exc_info:
        S.fp8x32  # pylint: disable=pointless-statement

    exc_msg = str(exc_info.value)
    expect = "module 'tvm.aipu.script' has no attribute 'fp8x32'"
    assert expect in exc_msg, f"\nExpect snippet:\n{expect}\n\nException Message:\n{exc_msg}\n"


if __name__ == "__main__":
    test_fwv_dtype_auto_gen()
    test_fwv_dtype_invalid()
