# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


dtype = "int32"
array_size = 1 << 10
vdtype = "int32x8"


@S.prim_func
def reassign_local(arr_buf: S.ptr(dtype, "global"), out_buf: S.ptr(dtype, "global")):
    a = 1
    if arr_buf[0] == 2:
        a = 2
    else:
        a = 3

    while a < 10:
        a += 2

    out_buf[0] = a


def test_reassign_local():
    n = 1
    a = rand(shape=(n,), dtype=dtype)
    gt_out = np.array((10 if a[0] == 2 else 11,), dtype)

    bm = BuildManager()
    ex = bm.build(reassign_local)

    py_out = np.empty(n, dtype)
    reassign_local(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


@S.prim_func
def reassign_local_vector(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.i32):
    scalar_init_va = S.int32x8(3)
    tuple_init_va = S.i32x8((0, 1, 2, 3, 4, 5, 6, 7))
    list_init_va = S.i32x8([0, 1, 2, 3, 4, 5, 6, 7])

    if c == 1:
        scalar_init_va += 1
        tuple_init_va += 1
    else:
        list_init_va = S.i32x8([1, 2, 3, 4, 5, 6, 7, 8])
        scalar_init_va += list_init_va
        tuple_init_va += 2

    b[0] = a[0] + scalar_init_va + tuple_init_va + list_init_va


def test_reassign_local_vector():
    n = 8
    a = np.zeros(n, dtype=dtype)
    gt_tmp0 = a + np.array([3] * n, dtype=dtype) + np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=dtype)
    gt_tmp1 = gt_tmp0 + np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=dtype) + 2
    gt_out = gt_tmp1 + np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=dtype)

    bm = BuildManager()
    ex = bm.build(reassign_local_vector)

    py_out = np.empty(n, dtype)
    reassign_local_vector(a, py_out, 2)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex.run(a, npu_out, 2)
    assert_allclose(npu_out, gt_out)


@S.prim_func
def reassign_param(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.i32):
    if a[0] == 2:
        c += 2
    else:
        c += 3

    while c < 10:
        c += 2

    b[0] = c


def test_reassign_param():
    n = 1
    a = rand(shape=(n,), dtype=dtype)
    gt_out = np.array((11 if a[0] == 2 else 10,), dtype)

    bm = BuildManager()
    ex = bm.build(reassign_param)

    py_out = np.empty(n, dtype)
    reassign_param(a, py_out, 3)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, npu_out, 3)
    assert_allclose(npu_out, gt_out)


@S.prim_func
def mul2_func(val: S.int32) -> S.int32:
    return val + val


@S.prim_func
def reassign_combine(arr: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
    sum_val = S.u32(0)
    sum_val += S.u32(arr[0] + arr[1])
    for i in range(array_size - 2):
        sum_val += S.u32(arr[i + 2])
    out[0] = mul2_func(sum_val)
    out[1] = sum_val


def test_reassign_combine():
    n = 2
    arr = rand(array_size, dtype)
    gt_out = np.empty(2, dtype)
    gt_out[1] = np.sum(arr)
    gt_out[0] = gt_out[1] * 2

    bm = BuildManager()
    ex = bm.build(reassign_combine)

    py_out = np.empty(n, dtype)
    reassign_combine(arr, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(arr, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_flexible_width_vector_reassign(lanes):
    @S.prim_func
    def fwv_reassign_func(inp0: S.ptr("int8", "global"), i32_in_zp: S.i32, out: S.ptr("int32", "global")):
        i8_data = S.vload(inp0, lanes=lanes)
        i32_data = S.cast(i8_data, "int32")
        i32_data = S.vadd(i32_data, i32_in_zp)
        S.vstore(i32_data, out)

    return fwv_reassign_func


def test_flexible_width_vector_reassign():
    n = 37
    a = rand(n, "int8")
    b = rand(1, "int32")

    gt_out = a.astype("int32") + b

    py_func = gen_flexible_width_vector_reassign(n)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype="int32")
    py_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype="int32")
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_reassign_local()
    test_reassign_local_vector()
    test_reassign_param()
    test_reassign_combine()
    test_flexible_width_vector_reassign()
