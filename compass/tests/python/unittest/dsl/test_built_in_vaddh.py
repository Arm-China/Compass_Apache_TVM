# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import get_range
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gt_vaddh(a, b, dtype):
    def np_addh(x, y):
        x = x.astype("int64")
        y = y.astype("int64")
        return np.clip(x + y, *get_range(dtype)).astype(dtype)

    n = len(a)
    c = np.zeros(n, dtype=dtype)
    for i in range(n // 2):
        c[i] = np_addh(a[2 * i], a[2 * i + 1])
        c[i + n // 2] = np_addh(b[2 * i], b[2 * i + 1])
    return c


def vaddh_generic(n, vdtype):
    @S.prim_func
    def vaddh_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        c[0] = S.vaddh(a[0], b[0])

    return vaddh_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_vaddh(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = gt_vaddh(a, b, dtype)

    prim_func = vaddh_generic(n, vdtype)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vaddh("int8")
