# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vclip_func(dtype, n, mask):
    @S.prim_func
    def vclip_func(x: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        out[0:n] = S.clip(x[0:n], 3, 8, mask)

    return vclip_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_scalar_flexible_width_vector(dtype):
    n = hw_native_vdtype(dtype).lanes + 3
    x = rand(n, dtype)
    mask = rand(n, "bool")
    gt_out = np.where(mask, np.clip(x, 3, 8), x)

    py_func = gen_vclip_func(dtype, n, mask)
    bm = BuildManager()
    ex = bm.build(py_func, name=f"vclip_{dtype}_flexible_width")

    py_out = np.empty(n, dtype)
    py_func(x, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(x, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_vmax_vmin_func(func_name, dtype, hw_lanes):
    sdot_func = {"maximum": S.max, "minimum": S.min}[func_name]
    lanes0 = hw_lanes // 2
    lanes1 = hw_lanes - 5
    lanes2 = hw_lanes + 3
    lanes3 = 2 * hw_lanes + 1
    lanes4 = 4 * hw_lanes

    @S.prim_func
    def vmax_vmin_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n / 2
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(sdot_func(va0, vb0), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0
        # 2. n - 5
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(sdot_func(va1, vb1), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1
        # 3. n + 3
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(sdot_func(va2, vb2), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2
        # 4. 2 * n + 1
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(sdot_func(va3, vb3), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes3, cur_b + lanes3, cur_out + lanes3
        # 5. 4 * n
        va4 = S.vload(cur_a, lanes=lanes4)
        vb4 = S.vload(cur_b, lanes=lanes4)
        S.vstore(sdot_func(va4, vb4), cur_out)

    return vmax_vmin_func


@pytest.mark.parametrize("func_name", ("maximum", "minimum"))
@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vmax_vmin(func_name, dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = int(8.5 * hw_lanes - 1)
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = getattr(np, func_name)(a, b)

    py_func = gen_vmax_vmin_func(func_name, dtype, hw_lanes)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_floating_classification_func(func_name, dtype, hw_lanes):
    sdot_func = {"isnan": S.isnan, "isinf": S.isinf, "isfinite": S.isfinite}[func_name]
    lanes0 = hw_lanes // 2
    lanes1 = hw_lanes - 5
    lanes2 = hw_lanes + 3
    lanes3 = 2 * hw_lanes + 1
    lanes4 = 4 * hw_lanes

    @S.prim_func
    def func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n / 2
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        mask0 = sdot_func(va0, vb0 > 0)
        S.vstore(vb0, cur_out, mask0)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0
        # 2. n - 5
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        mask1 = sdot_func(va1, vb1 > 0)
        S.vstore(vb1, cur_out, mask1)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1
        # 3. n + 3
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        mask2 = sdot_func(va2, vb2 > 0)
        S.vstore(vb2, cur_out, mask2)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2
        # 4. 2 * n + 1
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        mask3 = sdot_func(va3, vb3 > 0)
        S.vstore(vb3, cur_out, mask3)

        cur_a, cur_b, cur_out = cur_a + lanes3, cur_b + lanes3, cur_out + lanes3
        # 5. 4 * n
        va4 = S.vload(cur_a, lanes=lanes4)
        vb4 = S.vload(cur_b, lanes=lanes4)
        mask4 = sdot_func(va4, vb4 > 0)
        S.vstore(vb4, cur_out, mask4)

    return func


@pytest.mark.parametrize("func_name", ("isinf", "isnan", "isfinite"))
@pytest.mark.parametrize("dtype", ("float16", "float32"))
def test_floating_classification(func_name, dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = int(8.5 * hw_lanes - 1)

    a = rand(n, dtype)
    a[:33] = np.array(("nan",) * 13 + ("-inf", "inf") * 10, dtype)
    np.random.shuffle(a)

    b = rand(n, dtype)
    assert_mask = np.where(b > 0, getattr(np, func_name)(a), False)
    gt_out = b[:]

    py_func = gen_floating_classification_func(func_name, dtype, hw_lanes)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    assert_allclose(py_out[assert_mask], gt_out[assert_mask])

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out[assert_mask], gt_out[assert_mask])


if __name__ == "__main__":
    test_scalar_flexible_width_vector("int8")
    test_vmax_vmin("maximum", "int32")
    test_floating_classification("isinf", "float32")
