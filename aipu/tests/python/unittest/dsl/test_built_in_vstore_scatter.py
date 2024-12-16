# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
import pytest
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_vstore_scatter(vec_n, vdtype):
    @S.prim_func
    def vstore_scatter(
        value: S.ptr(vdtype, "global"),
        idx: S.ptr("u16", "global"),
        src: S.ptr(vdtype, "global"),
        out: S.ptr(vdtype, "global"),
    ):
        for i in range(vec_n):
            out[i] = src[i]

        vidx = S.vload(idx, lanes=vdtype.lanes)
        S.vstore_scatter(value[0], out, vidx)

    return vstore_scatter


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vstore_scatter(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = 64
    idx_n = vdtype.lanes
    value = rand(idx_n, dtype)
    idx = rand(idx_n, "uint16", high=n)
    src = rand(n, dtype)
    gt_out = src.copy()
    gt_out[idx] = value

    vstore_scatter = gen_vstore_scatter(n // idx_n, vdtype)
    bm = aipu.tir.BuildManager()
    ex = bm.build(vstore_scatter)

    py_out = np.empty(n, dtype=dtype)
    vstore_scatter(value, idx, src, py_out)
    testing.assert_allclose(py_out[idx], gt_out[idx])

    aipu_out = np.empty(n, dtype=dtype)
    ex(value, idx, src, aipu_out)
    testing.assert_allclose(aipu_out[idx], gt_out[idx])


def gen_vstore_scatter_composite(dtype):
    lanes = hw_native_vdtype(dtype).lanes

    @S.prim_func
    def vstore_scatter_composite(
        value: S.ptr(dtype, "global"),
        idx: S.ptr("u16", "global"),
        out: S.ptr(dtype, "global"),
        idx_n: S.i32,
    ):
        for i in range(idx_n // lanes):
            va = S.vload(value + i * lanes)
            indices = S.vload(idx + i * lanes, lanes=lanes)
            S.vstore_scatter(va, out, indices)

        tail = idx_n % lanes
        if tail != 0:
            mask = S.tail_mask(tail, lanes)
            indices_mask = S.tail_mask(tail, lanes)
            va = S.vload(value + idx_n - tail, lanes=lanes, mask=mask)
            indices = S.vload(idx + idx_n - tail, lanes=lanes, mask=indices_mask)
            S.vstore_scatter(va, out, indices, mask)

    return vstore_scatter_composite


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("idx_n", (64, 80))
def test_vstore_scatter_composite(dtype, idx_n):
    n = 1000
    value = rand(idx_n, dtype)
    idx = rand(idx_n, "uint16", high=n)
    gt_out = np.zeros(n, dtype)
    gt_out[idx] = value

    vstore_scatter = gen_vstore_scatter_composite(dtype)
    bm = aipu.tir.BuildManager()
    ex = bm.build(vstore_scatter)

    py_out = np.empty(n, dtype=dtype)
    vstore_scatter(value, idx, py_out, idx_n)
    testing.assert_allclose(py_out[idx], gt_out[idx])

    aipu_out = np.empty(n, dtype=dtype)
    ex(value, idx, aipu_out, idx_n)
    testing.assert_allclose(aipu_out[idx], gt_out[idx])


if __name__ == "__main__":
    test_vstore_scatter(dtype="int32")
    test_vstore_scatter(dtype="uint16")
    test_vstore_scatter(dtype="float32")
    test_vstore_scatter_composite(dtype="int8", idx_n=80)
    test_vstore_scatter_composite(dtype="uint16", idx_n=93)
    test_vstore_scatter_composite(dtype="float32", idx_n=66)
