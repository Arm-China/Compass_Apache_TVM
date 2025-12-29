# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_mask_concat():
    @S.prim_func
    def mask_concat(a0: S.ptr("int32", "global"), a1: S.ptr("int8x32", "global"), c: S.ptr("int8", "global")):
        va = S.vload(a0, lanes=32)  # 4 bool8 -> bool32
        mask = va > 0
        vc = S.vsel(a1[0], 0, mask)
        S.vstore(vc, c)

    return mask_concat


def gen_mask_split(case_id):
    @S.prim_func
    def mask_split1(a0: S.ptr("int8x32", "global"), a1: S.ptr("float32", "global"), c: S.ptr("float32", "global")):
        mask = a0[0] > 0
        vx = S.vload(a1, lanes=32)
        vc = S.vsel(vx, 0, mask)  # bool32 -> 4  bool8
        S.vstore(vc, c)

    @S.prim_func
    def mask_split2(a0: S.ptr("int8x32", "global"), a1: S.ptr("float16", "global"), c: S.ptr("float16", "global")):
        mask = a0[0] > 0
        vx = S.vload(a1, lanes=32)
        vc = S.vsel(vx, 0, mask)  # bool32 -> 2 bool16
        S.vstore(vc, c)

    @S.prim_func
    def mask_split3(a0: S.ptr("int16x16", "global"), a1: S.ptr("float32", "global"), c: S.ptr("float32", "global")):
        mask = a0[0] > 0
        vx = S.vload(a1, lanes=16)
        vc = S.vsel(vx, 0, mask)  # bool16 -> 2 bool8
        S.vstore(vc, c)

    if case_id == 1:
        return mask_split1
    elif case_id == 2:
        return mask_split2
    else:
        return mask_split3


def test_mask_concat():
    n = 32
    dtype = "int8"
    a0 = rand(n, "int32")
    a1 = rand(n, dtype)
    gt_out = np.where(a0 > 0, a1, [0] * n).astype(dtype)

    prim_func = gen_mask_concat()
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a0, a1, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a0, a1, npu_out)
    assert_allclose(npu_out, gt_out)


def test_mask_split():
    bm = BuildManager()

    # bool32 -> 4  bool8
    dtype = "float32"
    n = 32
    a0 = rand(n, "int8")
    a1 = rand(n, dtype)
    gt_out = np.where(a0 > 0, a1, [0] * n).astype(dtype)

    prim_func = gen_mask_split(1)
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a0, a1, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a0, a1, npu_out)
    assert_allclose(npu_out, gt_out)

    # bool32 -> 2 bool16
    dtype = "float16"
    a0 = rand(n, "int8")
    a1 = rand(n, dtype)
    gt_out = np.where(a0 > 0, a1, [0] * n).astype(dtype)

    prim_func = gen_mask_split(2)
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a0, a1, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a0, a1, npu_out)
    assert_allclose(npu_out, gt_out)

    # bool16 -> 2 bool8
    n = 16
    dtype = "float32"
    a0 = rand(n, "int16")
    a1 = rand(n, dtype)
    gt_out = np.where(a0 > 0, a1, [0] * n).astype(dtype)

    prim_func = gen_mask_split(3)
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a0, a1, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a0, a1, npu_out)
    assert_allclose(npu_out, gt_out)


@S.prim_func
def vconcat(
    a0: S.ptr("int8x32", "global"),
    a1: S.ptr("float32x32", "global"),
    c: S.ptr("float32x32", "global"),
):
    mask0 = a0[0] > 0
    mask1 = a1[0] > 0
    mask = S.vconcat([mask0, mask1], part="low")
    S.vstore(S.fp32x32(0), c)
    S.vstore(a1[0], c, mask=mask)


def test_mask_vconcat():
    n = 32
    a0 = rand(n, "int8")
    a1 = rand(n, "float32")
    gt_out = np.where(np.concatenate((a0[:16] > 0, a1[:16] > 0)), a1, 0)

    prim_func = vconcat
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, "float32")
    prim_func(a0, a1, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, "float32")
    ex(a0, a1, npu_out)
    assert_allclose(npu_out, gt_out)


@S.prim_func
def vload_gather(
    a: S.ptr("uint16x32", "global"),
    b: S.ptr("float32x32", "global"),
    c: S.ptr("float32x32", "global"),
):
    indices = a[0]
    mask = indices > 16
    c[0] = 0
    c[0] = S.vload_gather(b, indices, mask=mask)


def test_mask_vload_gather():
    n = 32
    a = rand(n, "uint16", high=32)
    b = rand(n, "float32")
    gt_out = np.where(a > 16, b[a], 0)

    prim_func = vload_gather
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, "float32")
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, "float32")
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


@S.prim_func
def vstore_scatter(
    a: S.ptr("uint16x32", "global"),
    b: S.ptr("float32x32", "global"),
    c: S.ptr("float32x32", "global"),
):
    indices = a[0]
    mask = indices > 16
    c[0] = 0
    S.vstore_scatter(b[0], c, indices, mask=mask)


def test_mask_vstore_scatter():
    n = 32
    a = rand(n, "uint16", high=32)
    b = rand(n, "float32")

    gt_out = np.zeros(n, "float32")
    mask = a > 16
    gt_out[a[mask]] = b[mask]

    prim_func = vstore_scatter
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, "float32")
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, "float32")
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_mask_concat()
    test_mask_split()
    test_mask_vconcat()
    test_mask_vload_gather()
    test_mask_vstore_scatter()
