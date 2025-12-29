# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import itertools
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vzip_func(dtype, hw_lanes):
    lanes0 = hw_lanes // 2
    lanes1 = hw_lanes - 5
    lanes2 = hw_lanes + 3
    lanes3 = 2 * hw_lanes + 1
    lanes4 = 4 * hw_lanes

    @S.prim_func
    def vzip_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n / 2
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(S.vzip(va0, vb0), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0 * 2
        # 2. n - 5
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(S.vzip(va1, vb1), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1 * 2
        # 3. n + 3
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(S.vzip(va2, vb2), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2 * 2
        # 4. 2 * n + 1
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(S.vzip(va3, vb3), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes3, cur_b + lanes3, cur_out + lanes3 * 2
        # 5. 4 * n
        va4 = S.vload(cur_a, lanes=lanes4)
        vb4 = S.vload(cur_b, lanes=lanes4)
        S.vstore(S.vzip(va4, vb4), cur_out)

    return vzip_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vzip(dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = int(8.5 * hw_lanes - 1)
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = np.array(list(itertools.chain.from_iterable(zip(a, b))), dtype=dtype)

    py_func = gen_vzip_func(dtype, hw_lanes)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n * 2, dtype=dtype)
    py_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n * 2, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_vzip_mask_func(dtype, hw_lanes):
    lanes0 = hw_lanes // 2
    lanes1 = hw_lanes - 5
    lanes2 = hw_lanes + 3
    lanes3 = 2 * hw_lanes + 1
    lanes4 = 4 * hw_lanes

    @S.prim_func
    def vzip_mask_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n / 2
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        vzip0 = S.vzip(va0, vb0)
        mask0 = S.vzip(va0 > 0, vb0 > 0)
        S.vstore(S.vsel(vzip0, 0, mask0), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0 * 2
        # 2. n - 5
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        vzip1 = S.vzip(va1, vb1)
        mask1 = S.vzip(va1 > 0, vb1 > 0)
        S.vstore(S.vsel(vzip1, 0, mask1), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1 * 2
        # 3. n + 3
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        vzip2 = S.vzip(va2, vb2)
        mask2 = S.vzip(va2 > 0, vb2 > 0)
        S.vstore(S.vsel(vzip2, 0, mask2), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2 * 2
        # 4. 2 * n + 1
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        vzip3 = S.vzip(va3, vb3)
        mask3 = S.vzip(va3 > 0, vb3 > 0)
        S.vstore(S.vsel(vzip3, 0, mask3), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes3, cur_b + lanes3, cur_out + lanes3 * 2
        # 5. 4 * n
        va4 = S.vload(cur_a, lanes=lanes4)
        vb4 = S.vload(cur_b, lanes=lanes4)
        vzip4 = S.vzip(va4, vb4)
        mask4 = S.vzip(va4 > 0, vb4 > 0)
        S.vstore(S.vsel(vzip4, 0, mask4), cur_out)

    return vzip_mask_func


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32"))
def test_vzip_mask(dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = int(8.5 * hw_lanes - 1)
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = np.maximum(list(itertools.chain.from_iterable(zip(a, b))), 0, dtype=dtype)

    py_func = gen_vzip_mask_func(dtype, hw_lanes)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n * 2, dtype=dtype)
    py_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n * 2, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vzip("int32")
    test_vzip_mask("int32")
