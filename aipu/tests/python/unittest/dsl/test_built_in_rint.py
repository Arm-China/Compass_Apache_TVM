# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_vrint_func(vdtype, mask):
    @S.prim_func
    def vrint_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global")):
        b[0] = S.rint(a[0], mask)

    return vrint_func


@pytest.mark.parametrize("dtype", ("float16", "float32"))
def test_vrint(dtype):
    vdtype = hw_native_vdtype(dtype)
    lane = vdtype.lanes

    a = rand(lane, dtype, low=-10, high=10)
    mask = rand(lane, "bool")
    gt_out = np.rint(a)

    f_rint = gen_vrint_func(vdtype, mask)
    bm = aipu.tir.BuildManager()
    ex = bm.build(f_rint)

    py_out = np.empty(lane, dtype)
    f_rint(a, py_out)
    testing.assert_allclose(py_out[mask], gt_out[mask])

    aipu_out = np.empty(lane, dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


def gen_rint_func(dtype):
    @S.prim_func
    def rint_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global")):
        b[0] = S.rint(1.234)
        b[0] = S.rint(a[0])

    return rint_func


@pytest.mark.parametrize("dtype", ("float16", "float32"))
def test_rint(dtype):
    lane = 1
    a = rand((lane,), dtype, low=-10, high=10)
    gt_out = np.rint(a)

    f_rint = gen_rint_func(dtype)
    bm = aipu.tir.BuildManager()
    ex = bm.build(f_rint)

    py_out = np.empty(lane, dtype)
    f_rint(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(lane, dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_vrint("float32")
    test_vrint("float16")
    test_rint("float32")
