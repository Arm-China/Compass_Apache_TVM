# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import rand


@S.prim_func
def explicit_tec_parallel(a: S.ptr("float16", "global"), b: S.ptr("float16", "global"), c: S.ptr("float16", "global")):
    for tid in S.tec_range(4):
        for j in S.vectorized(16):
            idx = tid * 16 + j
            c[idx] += a[idx] + b[idx]


def test_explicit_tec_parallel():
    a = np.array(range(64), dtype="float16")
    b = np.array(range(64), dtype="float16")

    bm = aipu.tir.BuildManager()
    ex = bm.build(explicit_tec_parallel)

    py_out = np.empty(64, "float16")
    gt_out = py_out + a + b
    explicit_tec_parallel(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(64, "float16")
    gt_out = aipu_out + a + b
    ex.run(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


@S.prim_func
def implicit_tec_parallel(a: S.ptr("float16", "global"), b: S.ptr("float16", "global"), c: S.ptr("float16", "global")):
    tid = S.get_local_id()
    for j in S.vectorized(16):
        c[tid * 16 + j] += a[tid * 16 + j] + b[tid * 16 + j]


def test_implicit_tec_parallel():
    a = np.array(range(64), dtype="float16")
    b = np.array(range(64), dtype="float16")

    bm = aipu.tir.BuildManager()
    ex = bm.build(implicit_tec_parallel)

    py_out = np.empty(64, "float16")
    gt_out = py_out + a + b
    implicit_tec_parallel(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(64, "float16")
    gt_out = aipu_out + a + b
    ex.run(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


@S.prim_func
def empty_return(a: S.ptr("float16", "global"), b: S.ptr("float16", "global"), c: S.ptr("float16", "global")):
    for tid in S.tec_range(4):
        if tid >= 2:
            c[tid * 16 : tid * 16 + 16] = 0
            return

        for j in S.vectorized(16):
            c[tid * 16 + j] = a[tid * 16 + j] + b[tid * 16 + j]


def test_empty_return():
    a = np.array(range(64), dtype="float16")
    b = np.array(range(64), dtype="float16")
    gt_out = a + b
    gt_out[32:] = 0

    bm = aipu.tir.BuildManager()
    ex = bm.build(empty_return)

    py_out = np.empty(64, "float16")
    empty_return(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(64, "float16")
    ex.run(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


@S.prim_func
def implicit_type_convertion(a: S.ptr("int32x8", "global"), b: S.ptr("int32x8", "global")):
    var = S.u32x8(2)
    var = S.u32x8(a[0])
    b[0] = var


def test_implicit_type_convertion():
    a = np.array([-1] * 8, dtype="int32")
    gt_out = a

    bm = aipu.tir.BuildManager()
    ex = bm.build(implicit_type_convertion)

    py_out = np.empty(8, "int32")
    implicit_type_convertion(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(8, "int32")
    ex.run(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


@S.prim_func
def if_int_stmt(a: S.ptr("int32x8", "global"), out: S.ptr("int32x8", "global")):
    var_a = a[0]
    tail_tmp16 = var_a[0] - 1
    if tail_tmp16:
        var_a[0] = 1
    out[0] = S.vabs(a[0])


def test_if_int_stmt():
    a = np.array(list(range(8)), dtype="int32")
    var_a = a[0] - 1
    if var_a:
        a[0] = 1
    gt_out = np.abs(a)

    bm = aipu.tir.BuildManager()
    ex = bm.build(if_int_stmt)

    py_out = np.empty(8, "int32")
    if_int_stmt(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(8, "int32")
    ex.run(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


@S.prim_func
def logical_and_mask(a: S.boolx32, b: S.boolx32, c: S.bool) -> S.boolx32:
    return S.vand(a, b)


@S.prim_func
def mask_as_sub_func_arg(a: S.ptr("int8x32", "global"), out: S.ptr("int8x32", "global")):
    mask_a = a[0] > 2
    mask_b = S.const_mask("16T16F")

    is_bigger = False
    mask_out = logical_and_mask(mask_a, mask_b, is_bigger)
    out[0] = S.vsel(a[0], 0, mask_out)


def test_mask_as_sub_func_arg():
    dtype, n = "int8", 32
    a = rand(n, dtype)
    gt_out = np.where(a > 2, a, 0)
    gt_out[n // 2 :] = 0

    bm = aipu.tir.BuildManager(disabled_pass=("AlignVectorWidthBySplit", "AlignVectorWidthByPad"))
    ex = bm.build(mask_as_sub_func_arg)

    py_out = np.empty(n, dtype)
    mask_as_sub_func_arg(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype)
    ex.run(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


@S.prim_func
def for_with_step(a: S.ptr("int8", "global"), out: S.ptr("int8", "global")):
    cnt = 0
    for i in range(0, 32, 2):
        out[cnt] = a[i]
        cnt += 1

    for i in range(1, 32, 2):
        out[cnt] = a[i]
        cnt += 1


def test_for_with_step():
    dtype, n = "int8", 32
    a = rand(n, dtype)
    gt_out = np.empty(n, dtype)
    gt_out[: n // 2] = a[::2]
    gt_out[n // 2 :] = a[1::2]

    bm = aipu.tir.BuildManager()
    ex = bm.build(for_with_step)

    py_out = np.empty(n, dtype)
    for_with_step(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype)
    ex.run(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_explicit_tec_parallel()
    test_implicit_tec_parallel()
    test_empty_return()
    test_implicit_type_convertion()
    test_if_int_stmt()
    test_mask_as_sub_func_arg()
    test_for_with_step()
