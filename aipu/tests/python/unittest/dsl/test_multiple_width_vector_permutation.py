# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import itertools
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_vconcat_all_func(dtype, hw_lanes, inp_num):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def vconcat2_all_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(S.vconcat((va0, vb0)), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0 * 2
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(S.vconcat((va1, vb1)), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1 * 2
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(S.vconcat((va2, vb2)), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2 * 2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(S.vconcat((va3, vb3)), cur_out)

    @S.prim_func
    def vconcat3_all_func(
        a: S.ptr(dtype, "global"),
        b: S.ptr(dtype, "global"),
        c: S.ptr(dtype, "global"),
        out: S.ptr(dtype, "global"),
    ):
        cur_a, cur_b, cur_c, cur_out = a, b, c, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        vc0 = S.vload(cur_c, lanes=lanes0)
        S.vstore(S.vconcat((va0, vb0, vc0)), cur_out)

        cur_a, cur_b, cur_c = cur_a + lanes0, cur_b + lanes0, cur_c + lanes0
        cur_out = cur_out + lanes0 * 3
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        vc1 = S.vload(cur_c, lanes=lanes1)
        S.vstore(S.vconcat((va1, vb1, vc1)), cur_out)

        cur_a, cur_b, cur_c = cur_a + lanes1, cur_b + lanes1, cur_c + lanes1
        cur_out = cur_out + lanes1 * 3
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        vc2 = S.vload(cur_c, lanes=lanes2)
        S.vstore(S.vconcat((va2, vb2, vc2)), cur_out)

        cur_a, cur_b, cur_c = cur_a + lanes2, cur_b + lanes2, cur_c + lanes2
        cur_out = cur_out + lanes2 * 3
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        vc3 = S.vload(cur_c, lanes=lanes3)
        S.vstore(S.vconcat((va3, vb3, vc3)), cur_out)

    @S.prim_func
    def vconcat4_all_func(
        a: S.ptr(dtype, "global"),
        b: S.ptr(dtype, "global"),
        c: S.ptr(dtype, "global"),
        d: S.ptr(dtype, "global"),
        out: S.ptr(dtype, "global"),
    ):
        cur_a, cur_b, cur_c, cur_d, cur_out = a, b, c, d, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        vc0 = S.vload(cur_c, lanes=lanes0)
        vd0 = S.vload(cur_d, lanes=lanes0)
        S.vstore(S.vconcat((va0, vb0, vc0, vd0)), cur_out)

        cur_a, cur_b, cur_c = cur_a + lanes0, cur_b + lanes0, cur_c + lanes0
        cur_d, cur_out = cur_d + lanes0, cur_out + lanes0 * 4
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        vc1 = S.vload(cur_c, lanes=lanes1)
        vd1 = S.vload(cur_d, lanes=lanes1)
        S.vstore(S.vconcat((va1, vb1, vc1, vd1)), cur_out)

        cur_a, cur_b, cur_c = cur_a + lanes1, cur_b + lanes1, cur_c + lanes1
        cur_d, cur_out = cur_d + lanes1, cur_out + lanes1 * 4
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        vc2 = S.vload(cur_c, lanes=lanes2)
        vd2 = S.vload(cur_d, lanes=lanes2)
        S.vstore(S.vconcat((va2, vb2, vc2, vd2)), cur_out)

        cur_a, cur_b, cur_c = cur_a + lanes2, cur_b + lanes2, cur_c + lanes2
        cur_d, cur_out = cur_d + lanes2, cur_out + lanes2 * 4
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        vc3 = S.vload(cur_c, lanes=lanes3)
        vd3 = S.vload(cur_d, lanes=lanes3)
        S.vstore(S.vconcat((va3, vb3, vc3, vd3)), cur_out)

    return locals()[f"vconcat{inp_num}_all_func"]


def get_concat_gt_out(inps, hw_lanes, part):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    if part == "all":
        return np.concatenate(
            (
                *[x[:lanes0] for x in inps],
                *[x[lanes0 : lanes0 + lanes1] for x in inps],
                *[x[lanes0 + lanes1 : lanes0 + lanes1 + lanes2] for x in inps],
                *[x[lanes0 + lanes1 + lanes2 :] for x in inps],
            )
        )

    if part == "low":
        return np.concatenate(
            (
                *[x[: lanes0 // 2] for x in inps],
                *[x[lanes0 : lanes0 + lanes1 // 2] for x in inps],
                *[x[lanes0 + lanes1 : lanes0 + lanes1 + lanes2 // 2] for x in inps],
                *[x[lanes0 + lanes1 + lanes2 : lanes0 + lanes1 + lanes2 + lanes3 // 2] for x in inps],
            )
        )

    if part == "high":
        return np.concatenate(
            (
                *[x[lanes0 // 2 : lanes0] for x in inps],
                *[x[lanes0 + lanes1 // 2 : lanes0 + lanes1] for x in inps],
                *[x[lanes0 + lanes1 + lanes2 // 2 : lanes0 + lanes1 + lanes2] for x in inps],
                *[x[lanes0 + lanes1 + lanes2 + lanes3 // 2 :] for x in inps],
            )
        )

    if part == "even":
        return np.concatenate(
            (
                *[x[:lanes0][::2] for x in inps],
                *[x[lanes0 : lanes0 + lanes1][::2] for x in inps],
                *[x[lanes0 + lanes1 : lanes0 + lanes1 + lanes2][::2] for x in inps],
                *[x[lanes0 + lanes1 + lanes2 :][::2] for x in inps],
            )
        )

    assert part == "odd", f'Unsupported part "{part}".'
    return np.concatenate(
        (
            *[x[:lanes0][1::2] for x in inps],
            *[x[lanes0 : lanes0 + lanes1][1::2] for x in inps],
            *[x[lanes0 + lanes1 : lanes0 + lanes1 + lanes2][1::2] for x in inps],
            *[x[lanes0 + lanes1 + lanes2 :][1::2] for x in inps],
        )
    )


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vconcat2_all(dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = get_concat_gt_out((a, b), hw_lanes, "all")

    py_func = gen_vconcat_all_func(dtype, hw_lanes, 2)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n * 2, dtype=dtype)
    py_func(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n * 2, dtype=dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


@pytest.mark.parametrize("dtype", ("uint8", "float16", "float32"))
def test_vconcat3_all(dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    c = rand(n, dtype)
    gt_out = get_concat_gt_out((a, b, c), hw_lanes, "all")

    py_func = gen_vconcat_all_func(dtype, hw_lanes, 3)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n * 3, dtype=dtype)
    py_func(a, b, c, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n * 3, dtype=dtype)
    ex(a, b, c, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


@pytest.mark.parametrize("dtype", ("int8", "uint16", "int32"))
def test_vconcat4_all(dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    c = rand(n, dtype)
    d = rand(n, dtype)
    gt_out = get_concat_gt_out((a, b, c, d), hw_lanes, "all")

    py_func = gen_vconcat_all_func(dtype, hw_lanes, 4)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n * 4, dtype=dtype)
    py_func(a, b, c, d, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n * 4, dtype=dtype)
    ex(a, b, c, d, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_vconcat_lheo_func(dtype, hw_lanes, part, inp_num):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def vconcat2_lheo_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(S.vconcat((va0, vb0), part), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(S.vconcat((va1, vb1), part), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(S.vconcat((va2, vb2), part), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(S.vconcat((va3, vb3), part), cur_out)

    @S.prim_func
    def vconcat3_lheo_func(
        a: S.ptr(dtype, "global"),
        b: S.ptr(dtype, "global"),
        c: S.ptr(dtype, "global"),
        out: S.ptr(dtype, "global"),
    ):
        cur_a, cur_b, cur_c, cur_out = a, b, c, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        vc0 = S.vload(cur_c, lanes=lanes0)
        S.vstore(S.vconcat((va0, vb0, vc0), part), cur_out)

        cur_a, cur_b, cur_c = cur_a + lanes0, cur_b + lanes0, cur_c + lanes0
        cur_out = cur_out + lanes0 * 3 // 2
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        vc1 = S.vload(cur_c, lanes=lanes1)
        S.vstore(S.vconcat((va1, vb1, vc1), part), cur_out)

        cur_a, cur_b, cur_c = cur_a + lanes1, cur_b + lanes1, cur_c + lanes1
        cur_out = cur_out + lanes1 * 3 // 2
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        vc2 = S.vload(cur_c, lanes=lanes2)
        S.vstore(S.vconcat((va2, vb2, vc2), part), cur_out)

        cur_a, cur_b, cur_c = cur_a + lanes2, cur_b + lanes2, cur_c + lanes2
        cur_out = cur_out + lanes2 * 3 // 2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        vc3 = S.vload(cur_c, lanes=lanes3)
        S.vstore(S.vconcat((va3, vb3, vc3), part), cur_out)

    @S.prim_func
    def vconcat4_lheo_func(
        a: S.ptr(dtype, "global"),
        b: S.ptr(dtype, "global"),
        c: S.ptr(dtype, "global"),
        d: S.ptr(dtype, "global"),
        out: S.ptr(dtype, "global"),
    ):
        cur_a, cur_b, cur_c, cur_d, cur_out = a, b, c, d, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        vc0 = S.vload(cur_c, lanes=lanes0)
        vd0 = S.vload(cur_d, lanes=lanes0)
        S.vstore(S.vconcat((va0, vb0, vc0, vd0), part), cur_out)

        cur_a, cur_b, cur_c = cur_a + lanes0, cur_b + lanes0, cur_c + lanes0
        cur_d, cur_out = cur_d + lanes0, cur_out + lanes0 * 4 // 2
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        vc1 = S.vload(cur_c, lanes=lanes1)
        vd1 = S.vload(cur_d, lanes=lanes1)
        S.vstore(S.vconcat((va1, vb1, vc1, vd1), part), cur_out)

        cur_a, cur_b, cur_c = cur_a + lanes1, cur_b + lanes1, cur_c + lanes1
        cur_d, cur_out = cur_d + lanes1, cur_out + lanes1 * 4 // 2
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        vc2 = S.vload(cur_c, lanes=lanes2)
        vd2 = S.vload(cur_d, lanes=lanes2)
        S.vstore(S.vconcat((va2, vb2, vc2, vd2), part), cur_out)

        cur_a, cur_b, cur_c = cur_a + lanes2, cur_b + lanes2, cur_c + lanes2
        cur_d, cur_out = cur_d + lanes2, cur_out + lanes2 * 4 // 2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        vc3 = S.vload(cur_c, lanes=lanes3)
        vd3 = S.vload(cur_d, lanes=lanes3)
        S.vstore(S.vconcat((va3, vb3, vc3, vd3), part), cur_out)

    return locals()[f"vconcat{inp_num}_lheo_func"]


@pytest.mark.parametrize("dtype", ("uint8", "float16", "float32"))
@pytest.mark.parametrize("part", ("low", "high", "even", "odd"))
def test_vconcat2_lheo(part, dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = get_concat_gt_out((a, b), hw_lanes, part)

    py_func = gen_vconcat_lheo_func(dtype, hw_lanes, part, 2)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


@pytest.mark.parametrize("dtype", ("int8", "uint16", "int32"))
@pytest.mark.parametrize("part", ("low", "high", "even", "odd"))
def test_vconcat3_lheo(part, dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    c = rand(n, dtype)
    gt_out = get_concat_gt_out((a, b, c), hw_lanes, part)

    py_func = gen_vconcat_lheo_func(dtype, hw_lanes, part, 3)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n * 3 // 2, dtype=dtype)
    py_func(a, b, c, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n * 3 // 2, dtype=dtype)
    ex(a, b, c, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("part", ("low", "high", "even", "odd"))
def test_vconcat4_lheo(part, dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    c = rand(n, dtype)
    d = rand(n, dtype)
    gt_out = get_concat_gt_out((a, b, c, d), hw_lanes, part)

    py_func = gen_vconcat_lheo_func(dtype, hw_lanes, part, 4)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n * 4 // 2, dtype=dtype)
    py_func(a, b, c, d, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n * 4 // 2, dtype=dtype)
    ex(a, b, c, d, aipu_out)
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


def gen_vrevs_func(dtype, hw_lanes):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def vrevs_func(a: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_out = a, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        S.vstore(S.vrevs(va0), cur_out)

        cur_a, cur_out = cur_a + lanes0, cur_out + lanes0
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        S.vstore(S.vrevs(va1), cur_out)

        cur_a, cur_out = cur_a + lanes1, cur_out + lanes1
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        S.vstore(S.vrevs(va2), cur_out)

        cur_a, cur_out = cur_a + lanes2, cur_out + lanes2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        S.vstore(S.vrevs(va3), cur_out)

    return vrevs_func


def get_vrevs_gt_out(a, hw_lanes):
    return np.concatenate(
        (
            a[:hw_lanes][::-1],
            a[hw_lanes : 3 * hw_lanes][::-1],
            a[3 * hw_lanes : 6 * hw_lanes][::-1],
            a[6 * hw_lanes :][::-1],
        )
    )


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vrevs(dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    a = rand(n, dtype)
    gt_out = get_vrevs_gt_out(a, hw_lanes)

    py_func = gen_vrevs_func(dtype, hw_lanes)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_vsldl_func(dtype, hw_lanes):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def vsldl_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(S.vsldl(va0, vb0, lanes0 - 2), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(S.vsldl(va1, vb1, lanes0), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(S.vsldl(va2, vb2, lanes2 - 2), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(S.vsldl(va3, vb3, lanes2 - 2), cur_out)

    return vsldl_func


def get_vsldl_gt_out(a, b, hw_lanes):
    lanes0 = hw_lanes
    lanes2 = 3 * hw_lanes

    def _get_single_vsldl(x, y, shift):
        return np.concatenate([x[shift:], y[:shift]])

    return np.concatenate(
        (
            _get_single_vsldl(a[:lanes0], b[:lanes0], lanes0 - 2),
            _get_single_vsldl(a[lanes0 : 3 * lanes0], b[lanes0 : 3 * lanes0], lanes0),
            _get_single_vsldl(a[3 * lanes0 : 6 * lanes0], b[3 * lanes0 : 6 * lanes0], lanes2 - 2),
            _get_single_vsldl(a[6 * lanes0 :], b[6 * lanes0 :], lanes2 - 2),
        )
    )


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vsldl(dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = get_vsldl_gt_out(a, b, hw_lanes)

    py_func = gen_vsldl_func(dtype, hw_lanes)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_vreplic_func(dtype, hw_lanes, idx0, idx1, idx2, idx3):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def vreplic_func(a: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_out = a, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        S.vstore(S.vreplic(va0, idx0), cur_out)

        cur_a, cur_out = cur_a + lanes0, cur_out + lanes0
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        S.vstore(S.vreplic(va1, idx1), cur_out)

        cur_a, cur_out = cur_a + lanes1, cur_out + lanes1
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        S.vstore(S.vreplic(va2, idx2), cur_out)

        cur_a, cur_out = cur_a + lanes2, cur_out + lanes2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        S.vstore(S.vreplic(va3, idx3), cur_out)

    return vreplic_func


def get_vreplic_gt_out(a, hw_lanes, idx0, idx1, idx2, idx3):
    return np.concatenate(
        (
            [a[:hw_lanes][idx0]] * hw_lanes,
            [a[hw_lanes : 3 * hw_lanes][idx1]] * (2 * hw_lanes),
            [a[3 * hw_lanes : 6 * hw_lanes][idx2]] * (3 * hw_lanes),
            [a[6 * hw_lanes :][idx3]] * (4 * hw_lanes),
        )
    )


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vreplic(dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    idx0 = int(rand(1, "int8", low=0, high=hw_lanes - 1))
    idx1 = int(rand(1, "int8", low=0, high=2 * hw_lanes - 1))
    idx2 = int(rand(1, "int8", low=0, high=3 * hw_lanes - 1))
    idx3 = int(rand(1, "int8", low=0, high=4 * hw_lanes - 1))
    n = 10 * hw_lanes
    a = rand(n, dtype)
    gt_out = get_vreplic_gt_out(a, hw_lanes, idx0, idx1, idx2, idx3)

    py_func = gen_vreplic_func(dtype, hw_lanes, idx0, idx1, idx2, idx3)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def test_vreplic_non_constant_idx(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func(a: S.ptr("int32", "global"), idx: S.i32, out: S.ptr("int32", "global")):
            va = S.vload(a, lanes=16)
            S.vstore(S.vreplic(va, idx), out)

        aipu.tir.BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = 'The arg "index" expects an integer constant value.'
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def gen_vshfl_func(dtype, hw_lanes, shift0, shift1, shift2, shift3):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def vshfl_func(a: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_out = a, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        S.vstore(S.vshfl(va0, shift0), cur_out)

        cur_a, cur_out = cur_a + lanes0, cur_out + lanes0
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        S.vstore(S.vshfl(va1, shift1), cur_out)

        cur_a, cur_out = cur_a + lanes1, cur_out + lanes1
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        S.vstore(S.vshfl(va2, shift2), cur_out)

        cur_a, cur_out = cur_a + lanes2, cur_out + lanes2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        S.vstore(S.vshfl(va3, shift3), cur_out)

    return vshfl_func


def get_vshfl_gt_out(a, hw_lanes, shift0, shift1, shift2, shift3):
    return np.concatenate(
        (
            a[shift0:hw_lanes],
            a[:shift0],
            a[shift1 + hw_lanes : 3 * hw_lanes],
            a[hw_lanes : hw_lanes + shift1],
            a[shift2 + 3 * hw_lanes : 6 * hw_lanes],
            a[3 * hw_lanes : 3 * hw_lanes + shift2],
            a[shift3 + 6 * hw_lanes :],
            a[6 * hw_lanes : 6 * hw_lanes + shift3],
        )
    )


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vshfl(dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    a = rand(n, dtype)
    shift0 = rand(1, "int32", low=0, high=hw_lanes, return_python_type=True)
    shift1 = rand(1, "int32", low=0, high=2 * hw_lanes, return_python_type=True)
    shift2 = 2 * hw_lanes
    shift3 = rand(1, "int32", low=0, high=4 * hw_lanes, return_python_type=True)
    gt_out = get_vshfl_gt_out(a, hw_lanes, shift0, shift1, shift2, shift3)

    py_func = gen_vshfl_func(dtype, hw_lanes, shift0, shift1, shift2, shift3)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_vconcat2_all("int32")
    test_vconcat3_all("int32")
    test_vconcat4_all("int32")
    test_vconcat2_lheo("low", "int32")
    test_vconcat2_lheo("high", "int32")
    test_vconcat2_lheo("even", "int32")
    test_vconcat2_lheo("odd", "int32")
    test_vconcat3_lheo("low", "int32")
    test_vconcat3_lheo("high", "int32")
    test_vconcat3_lheo("even", "int32")
    test_vconcat3_lheo("odd", "int32")
    test_vconcat4_lheo("low", "int32")
    test_vconcat4_lheo("high", "int32")
    test_vconcat4_lheo("even", "int32")
    test_vconcat4_lheo("odd", "int32")
    test_vzip_lheo("low", "int8")
    test_vzip_lheo("even", "int16")
    test_vzip_mask_lheo("high", "int32")
    test_vzip_mask_lheo("odd", "int32")
    test_vrevs("int32")
    test_vsldl("int32")
    test_vreplic("int32")
    test_vreplic_non_constant_idx(None)
    test_vshfl("int32")
