# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def vmul_generic(vdtype, mask):
    @S.prim_func
    def vmul_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        c[0] = S.vmul(a[0], b[0], mask)

    return vmul_func


@pytest.mark.parametrize("dtype", ("int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vmul(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    mask = rand(n, "bool")
    gt_out = np.multiply(a, b, where=mask, dtype=dtype)

    py_func = vmul_generic(vdtype, mask)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    testing.assert_allclose(py_out[mask], gt_out[mask])

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


def vmul_with_r_generic(vdtype, mask):
    @S.prim_func
    def vmul_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        c[0] = S.vmul(a[0], b[0], mask, r=a[0])

    return vmul_func


def get_gt_out(a, b, mask, dtype):
    gt_out = np.multiply(a, b, where=mask, dtype=dtype)
    for idx, ele in enumerate(mask):
        if not ele:
            gt_out[idx] = a[idx]
    return gt_out


@pytest.mark.parametrize("dtype", ("int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vmul_with_r(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    mask = rand(n, "bool")
    gt_out = get_gt_out(a, b, mask, dtype)

    py_func = vmul_with_r_generic(vdtype, mask)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_vmul("int16")
    test_vmul_with_r("int16")
