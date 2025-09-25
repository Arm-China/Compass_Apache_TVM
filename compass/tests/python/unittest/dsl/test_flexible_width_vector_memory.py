# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vload_gather_func(dtype, indices_dtype, hw_lanes):
    lanes0 = hw_lanes // 2
    lanes1 = hw_lanes - 5
    lanes2 = hw_lanes + 3
    lanes3 = 2 * hw_lanes + 1
    lanes4 = 4 * hw_lanes

    @S.prim_func
    def vload_gather_func(a: S.ptr(dtype, "global"), b: S.ptr(indices_dtype, "global"), out: S.ptr(dtype, "global")):
        cur_b, cur_out = b, out
        # 1. n / 2
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(S.vload_gather(a, vb0), cur_out)

        cur_b, cur_out = cur_b + lanes0, cur_out + lanes0
        # 2. n - 5
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(S.vload_gather(a, vb1), cur_out)

        cur_b, cur_out = cur_b + lanes1, cur_out + lanes1
        # 3. n + 3
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(S.vload_gather(a, vb2), cur_out)

        cur_b, cur_out = cur_b + lanes2, cur_out + lanes2
        # 4. 2 * n + 1
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(S.vload_gather(a, vb3), cur_out)

        cur_b, cur_out = cur_b + lanes3, cur_out + lanes3
        # 5. 4 * n
        vb4 = S.vload(cur_b, lanes=lanes4)
        S.vstore(S.vload_gather(a, vb4), cur_out)

    return vload_gather_func


@pytest.mark.parametrize("indices_dtype", ("int16", "uint16"))
@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vload_gather(dtype, indices_dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = int(8.5 * hw_lanes - 1)
    a = rand(n, dtype)
    b = rand(n, indices_dtype, low=0, high=n)
    gt_out = a[b]

    py_func = gen_vload_gather_func(dtype, indices_dtype, hw_lanes)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_vstore_scatter_func(dtype, indices_dtype, hw_lanes):
    lanes0 = hw_lanes // 2
    lanes1 = hw_lanes - 5
    lanes2 = hw_lanes + 3
    lanes3 = 2 * hw_lanes + 1
    lanes4 = 4 * hw_lanes

    @S.prim_func
    def vstore_scatter_func(a: S.ptr(dtype, "global"), b: S.ptr(indices_dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b = a, b
        # 1. n / 2
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore_scatter(va0, out, vb0)

        cur_a, cur_b = cur_a + lanes0, cur_b + lanes0
        # 2. n - 5
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore_scatter(va1, out, vb1)

        cur_a, cur_b = cur_a + lanes1, cur_b + lanes1
        # 3. n + 3
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore_scatter(va2, out, vb2)

        cur_a, cur_b = cur_a + lanes2, cur_b + lanes2
        # 4. 2 * n + 1
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore_scatter(va3, out, vb3)

        cur_a, cur_b = cur_a + lanes3, cur_b + lanes3
        # 5. 4 * n
        va4 = S.vload(cur_a, lanes=lanes4)
        vb4 = S.vload(cur_b, lanes=lanes4)
        S.vstore_scatter(va4, out, vb4)

    return vstore_scatter_func


@pytest.mark.parametrize("indices_dtype", ("int16", "uint16"))
@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vstore_scatter(dtype, indices_dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = int(8.5 * hw_lanes - 1)
    a = rand(n, dtype)
    b = rand(n, indices_dtype, low=0, high=n)
    gt_out = np.zeros(n, dtype)
    gt_out[b] = a

    py_func = gen_vstore_scatter_func(dtype, indices_dtype, hw_lanes)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    assert_allclose(py_out[b], gt_out[b])

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out[b], gt_out[b])


if __name__ == "__main__":
    test_vload_gather("int32", "int16")
    test_vstore_scatter("int8", "int16")
