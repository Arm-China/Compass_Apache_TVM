# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def vmaxh_generic(vdtype):
    @S.prim_func
    def vmaxh_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        c[0] = S.vmaxh(a[0], b[0])

    return vmaxh_func


def vminh_generic(vdtype):
    @S.prim_func
    def vminh_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        c[0] = S.vminh(a[0], b[0])

    return vminh_func


def gt_vmaxh(a, b):
    n = len(a)
    c = np.zeros(n, dtype=a.dtype)
    for i in range(n // 2):
        c[i] = max(a[2 * i], a[2 * i + 1])
        c[i + n // 2] = max(b[2 * i], b[2 * i + 1])
    return c


def gt_vminh(a, b):
    n = len(a)
    c = np.zeros(n, dtype=a.dtype)
    for i in range(n // 2):
        c[i] = min(a[2 * i], a[2 * i + 1])
        c[i + n // 2] = min(b[2 * i], b[2 * i + 1])
    return c


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_all_vmaxh(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = gt_vmaxh(a, b)

    prim_func = vmaxh_generic(vdtype)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_all_vminh(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = gt_vminh(a, b)

    prim_func = vminh_generic(vdtype)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


def test_fail_diff_type(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func(a: S.ptr("i8x32", "global"), b: S.ptr("u8x32", "global"), c: S.ptr("i8x32", "global")):
            c[0] = S.vmaxh(a[0], b[0])

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = "The sign of operands is different"
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


if __name__ == "__main__":
    test_all_vmaxh("uint32")
    test_all_vminh("uint32")
    test_fail_diff_type(None)
