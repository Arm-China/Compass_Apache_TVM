# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu, get_range
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_vsubh(vdtype):
    @S.prim_func
    def vsubh_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        c[0] = S.vsubh(a[0], b[0])

    return vsubh_func


def gt_vsubh(a, b, dtype):
    def np_subh(x, y):
        x = x.astype("int64")
        y = y.astype("int64")
        return np.clip(x - y, *get_range(dtype)).astype(dtype)

    n = len(a)
    c = np.zeros(n, dtype=a.dtype)
    for i in range(n // 2):
        c[i] = np_subh(a[2 * i], a[2 * i + 1])
        c[i + n // 2] = np_subh(b[2 * i], b[2 * i + 1])
    return c


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_vsubh(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = gt_vsubh(a, b, dtype)

    prim_func = gen_vsubh(vdtype)
    bm = aipu.tir.BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_vsubh("int8")
