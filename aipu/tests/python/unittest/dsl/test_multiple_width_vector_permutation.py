# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import itertools
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_vconcat_all_func(dtype, hw_lanes):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def vconcat_all_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(S.vconcat(va0, vb0), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0 * 2
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(S.vconcat(va1, vb1), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1 * 2
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(S.vconcat(va2, vb2), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2 * 2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(S.vconcat(va3, vb3), cur_out)

    return vconcat_all_func


def get_gt_out(a, b, hw_lanes, part):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    if part == "all":
        return np.concatenate(
            (
                a[:lanes0],
                b[:lanes0],
                a[lanes0 : lanes0 + lanes1],
                b[lanes0 : lanes0 + lanes1],
                a[lanes0 + lanes1 : lanes0 + lanes1 + lanes2],
                b[lanes0 + lanes1 : lanes0 + lanes1 + lanes2],
                a[lanes0 + lanes1 + lanes2 :],
                b[lanes0 + lanes1 + lanes2 :],
            )
        )

    if part == "low":
        return np.concatenate(
            (
                a[: lanes0 // 2],
                b[: lanes0 // 2],
                a[lanes0 : lanes0 + lanes1 // 2],
                b[lanes0 : lanes0 + lanes1 // 2],
                a[lanes0 + lanes1 : lanes0 + lanes1 + lanes2 // 2],
                b[lanes0 + lanes1 : lanes0 + lanes1 + lanes2 // 2],
                a[lanes0 + lanes1 + lanes2 : lanes0 + lanes1 + lanes2 + lanes3 // 2],
                b[lanes0 + lanes1 + lanes2 : lanes0 + lanes1 + lanes2 + lanes3 // 2],
            )
        )

    if part == "high":
        return np.concatenate(
            (
                a[lanes0 // 2 : lanes0],
                b[lanes0 // 2 : lanes0],
                a[lanes0 + lanes1 // 2 : lanes0 + lanes1],
                b[lanes0 + lanes1 // 2 : lanes0 + lanes1],
                a[lanes0 + lanes1 + lanes2 // 2 : lanes0 + lanes1 + lanes2],
                b[lanes0 + lanes1 + lanes2 // 2 : lanes0 + lanes1 + lanes2],
                a[lanes0 + lanes1 + lanes2 + lanes3 // 2 :],
                b[lanes0 + lanes1 + lanes2 + lanes3 // 2 :],
            )
        )

    if part == "even":
        return np.concatenate(
            (
                a[:lanes0][::2],
                b[:lanes0][::2],
                a[lanes0 : lanes0 + lanes1][::2],
                b[lanes0 : lanes0 + lanes1][::2],
                a[lanes0 + lanes1 : lanes0 + lanes1 + lanes2][::2],
                b[lanes0 + lanes1 : lanes0 + lanes1 + lanes2][::2],
                a[lanes0 + lanes1 + lanes2 :][::2],
                b[lanes0 + lanes1 + lanes2 :][::2],
            )
        )

    assert part == "odd", f'Unsupported part "{part}".'
    return np.concatenate(
        (
            a[:lanes0][1::2],
            b[:lanes0][1::2],
            a[lanes0 : lanes0 + lanes1][1::2],
            b[lanes0 : lanes0 + lanes1][1::2],
            a[lanes0 + lanes1 : lanes0 + lanes1 + lanes2][1::2],
            b[lanes0 + lanes1 : lanes0 + lanes1 + lanes2][1::2],
            a[lanes0 + lanes1 + lanes2 :][1::2],
            b[lanes0 + lanes1 + lanes2 :][1::2],
        )
    )


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vconcat_all(dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = get_gt_out(a, b, hw_lanes, "all")

    py_func = gen_vconcat_all_func(dtype, hw_lanes)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n * 2, dtype=dtype)
    py_func(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n * 2, dtype=dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_vconcat_lheo_func(dtype, hw_lanes, part):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def vconcat_lheo_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(S.vconcat(va0, vb0, part), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(S.vconcat(va1, vb1, part), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(S.vconcat(va2, vb2, part), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(S.vconcat(va3, vb3, part), cur_out)

    return vconcat_lheo_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("part", ("low", "high", "even", "odd"))
def test_vconcat_lheo(part, dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = get_gt_out(a, b, hw_lanes, part)

    py_func = gen_vconcat_lheo_func(dtype, hw_lanes, part)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def get_vzip_gt_out_single(x, y, part, is_xy_mask):
    if is_xy_mask:
        x, y = x > 0, y > 0

    count = len(x) // 2
    if part == "low":
        gt_out = itertools.chain.from_iterable(zip(x[:count], y[:count]))
    elif part == "high":
        gt_out = itertools.chain.from_iterable(zip(x[count:], y[count:]))
    elif part == "even":
        gt_out = itertools.chain.from_iterable(zip(x[::2], y[::2]))
    else:
        assert part == "odd"
        gt_out = itertools.chain.from_iterable(zip(x[1::2], y[1::2]))

    return np.array(list(gt_out), dtype=x.dtype)


def get_vzip_gt_out(a, b, part, is_xy_mask, hw_lanes):
    out = np.zeros(10 * hw_lanes, a.dtype)
    cur_a = a[:hw_lanes]
    cur_b = b[:hw_lanes]
    out[:hw_lanes] = get_vzip_gt_out_single(cur_a, cur_b, part, is_xy_mask)
    cur_a = a[hw_lanes : hw_lanes * 3]
    cur_b = b[hw_lanes : hw_lanes * 3]
    out[hw_lanes : hw_lanes * 3] = get_vzip_gt_out_single(cur_a, cur_b, part, is_xy_mask)
    cur_a = a[hw_lanes * 3 : hw_lanes * 6]
    cur_b = b[hw_lanes * 3 : hw_lanes * 6]
    out[hw_lanes * 3 : hw_lanes * 6] = get_vzip_gt_out_single(cur_a, cur_b, part, is_xy_mask)
    cur_a = a[hw_lanes * 6 :]
    cur_b = b[hw_lanes * 6 :]
    out[hw_lanes * 6 :] = get_vzip_gt_out_single(cur_a, cur_b, part, is_xy_mask)
    return out


def gen_vzip_lheo_func(dtype, hw_lanes, part):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def vzip_lheo_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(S.vzip(va0, vb0, part), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(S.vzip(va1, vb1, part), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(S.vzip(va2, vb2, part), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(S.vzip(va3, vb3, part), cur_out)

    return vzip_lheo_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("part", ("low", "high", "even", "odd"))
def test_vzip_lheo(part, dtype):
    vdtype = hw_native_vdtype(dtype)
    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = get_vzip_gt_out(a, b, part, False, hw_lanes)

    py_func = gen_vzip_lheo_func(dtype, hw_lanes, part)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_vzip_mask_lheo_func(dtype, hw_lanes, part):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def vzip_mask_lheo_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        mask_a0 = va0 > 0
        mask_b0 = vb0 > 0
        mask_out0 = S.vzip(mask_a0, mask_b0, part)
        S.vstore(S.vsel(va0, vb0, mask_out0), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        mask_a1 = va1 > 0
        mask_b1 = vb1 > 0
        mask_out1 = S.vzip(mask_a1, mask_b1, part)
        S.vstore(S.vsel(va1, vb1, mask_out1), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        mask_a2 = va2 > 0
        mask_b2 = vb2 > 0
        mask_out2 = S.vzip(mask_a2, mask_b2, part)
        S.vstore(S.vsel(va2, vb2, mask_out2), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        mask_a3 = va3 > 0
        mask_b3 = vb3 > 0
        mask_out3 = S.vzip(mask_a3, mask_b3, part)
        S.vstore(S.vsel(va3, vb3, mask_out3), cur_out)

    return vzip_mask_lheo_func


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32"))
@pytest.mark.parametrize("part", ("low", "high", "even", "odd"))
def test_vzip_mask_lheo(part, dtype):
    vdtype = hw_native_vdtype(dtype)
    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_tmp = get_vzip_gt_out(a, b, part, True, hw_lanes)
    gt_out = np.where(gt_tmp, a, b)

    f_vzip = gen_vzip_mask_lheo_func(dtype, hw_lanes, part)
    bm = aipu.tir.BuildManager()
    ex = bm.build(f_vzip)

    py_out = np.empty(n, dtype)
    f_vzip(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_vconcat_all("int32")
    test_vconcat_lheo("low", "int32")
    test_vconcat_lheo("high", "int32")
    test_vconcat_lheo("even", "int32")
    test_vconcat_lheo("odd", "int32")
    test_vzip_lheo("low", "int8")
    test_vzip_lheo("even", "int16")
    test_vzip_mask_lheo("high", "int32")
    test_vzip_mask_lheo("odd", "int32")
