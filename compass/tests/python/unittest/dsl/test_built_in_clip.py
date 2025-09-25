# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import DataType
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vector_with_scalar_func(vdtype, min_val, max_val, mask):
    @S.prim_func
    def vclip_scalar_minmax_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global")):
        b[0] = S.clip(a[0], min_val, max_val, mask)

    return vclip_scalar_minmax_func


DTYPE_TUPLE = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32", "bfloat16")


@pytest.mark.parametrize("dtype", DTYPE_TUPLE)
def test_vector_with_scalar_minmax(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    min_val = np.sort(a)[2].tolist()
    max_val = np.sort(a)[n - 2].tolist()
    mask = rand(n, "bool")
    gt_out = np.where(mask, np.clip(a, min_val, max_val), a).astype(dtype)

    prim_func = gen_vector_with_scalar_func(vdtype, min_val, max_val, mask)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_vector_func(vdtype, mask):
    @S.prim_func
    def vclip_vector_minmax_func(
        a: S.ptr(vdtype, "global"),
        min_val: S.ptr(vdtype, "global"),
        max_val: S.ptr(vdtype, "global"),
        b: S.ptr(vdtype, "global"),
    ):
        b[0] = S.clip(a[0], min_val[0], max_val[0], mask)

    return vclip_vector_minmax_func


@pytest.mark.parametrize("dtype", DTYPE_TUPLE)
def test_vector(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    sorted_a = np.sort(a)
    min_val = np.repeat(sorted_a[1:3], n // 2)
    max_val = np.repeat(sorted_a[5:7], n // 2)
    mask = rand(n, "bool")
    gt_out = np.where(mask, np.clip(a, min_val, max_val), a).astype(dtype)

    prim_func = gen_vector_func(vdtype, mask)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, min_val, max_val, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, min_val, max_val, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_clip(dtype, min_val, max_val):
    @S.prim_func
    def clip_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global")):
        b[0] = S.clip(a[0], min_val, max_val)

    return clip_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("minmax_dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_scalar(dtype, minmax_dtype):
    vdtype = hw_native_vdtype(dtype)
    minmax_dtype = DataType(minmax_dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    if vdtype.is_integer and minmax_dtype.is_float:
        pytest.skip("Unuspported integer x with float min or max value.")

    print(f"test_clip_{dtype}_{minmax_dtype}")
    min_val = np.sort(a)[2].tolist()
    max_val = np.sort(a)[n - 2].tolist()
    gt_out = np.clip(a, min_val, max_val).astype(dtype)

    prim_func = gen_clip(dtype, min_val, max_val)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, py_out)
    assert_allclose(py_out[0], gt_out[0])

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out[0], gt_out[0])


if __name__ == "__main__":
    test_vector("int16")
    test_vector_with_scalar_minmax("uint32")
    test_scalar("int8", minmax_dtype="int16")
