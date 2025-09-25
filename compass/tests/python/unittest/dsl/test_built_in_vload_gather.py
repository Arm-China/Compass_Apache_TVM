# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
import pytest
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vload_gather(vdtype):
    @S.prim_func
    def vload_gather(a: S.ptr(vdtype, "global"), b: S.ptr("u16", "global"), c: S.ptr(vdtype, "global")):
        indices = S.vload(b, lanes=vdtype.lanes)
        c[0] = S.vload_gather(a, indices)

    return vload_gather


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vload_gather(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = 64
    out_n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(out_n, "uint16", high=n)
    gt_out = a[b]

    vload_gather = gen_vload_gather(vdtype)
    bm = BuildManager()
    ex = bm.build(vload_gather)

    py_out = np.empty(out_n, dtype=dtype)
    vload_gather(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(out_n, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_vload_gather_composite(dtype):
    lanes = hw_native_vdtype(dtype).lanes

    @S.prim_func
    def vload_gather_composite(
        a: S.ptr(dtype, "global"),
        b: S.ptr("u16", "global"),
        c: S.ptr(dtype, "global"),
        idx_n: S.i32,
    ):
        for i in range(idx_n // lanes):
            indices = S.vload(b + i * lanes, lanes=lanes)
            vc = S.vload_gather(a, indices)
            S.vstore(vc, c + i * lanes)

        tail = idx_n % lanes
        if tail != 0:
            mask = S.tail_mask(tail, lanes)
            indices_mask = S.tail_mask(tail, lanes)
            indices = S.vload(b + idx_n - tail, lanes=lanes, mask=indices_mask)
            vc = S.vload_gather(a, indices, mask=mask)
            S.vstore(vc, c + idx_n - tail, mask)

    return vload_gather_composite


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("idx_n", (64, 80))
def test_vload_gather_composite(dtype, idx_n):
    n = 1000
    a = rand(n, dtype)
    b = rand(idx_n, "uint16", high=n)
    gt_out = a[b]

    vload_gather = gen_vload_gather_composite(dtype)
    bm = BuildManager()
    ex = bm.build(vload_gather)

    py_out = np.empty(idx_n, dtype=dtype)
    vload_gather(a, b, py_out, idx_n)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(idx_n, dtype=dtype)
    ex(a, b, npu_out, idx_n)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vload_gather(dtype="int8")
    test_vload_gather(dtype="uint16")
    test_vload_gather(dtype="float32")
    test_vload_gather_composite(dtype="int8", idx_n=80)
    test_vload_gather_composite(dtype="uint16", idx_n=93)
    test_vload_gather_composite(dtype="float32", idx_n=66)
