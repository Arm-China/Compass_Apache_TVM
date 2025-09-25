# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, schedule
from tvm.compass.dsl.testing import rand, assert_allclose


dtype = "int8"


@S.prim_func
def add_schedule(A: S.ptr(dtype, "global"), B: S.ptr(dtype, "global"), C: S.ptr(dtype, "global"), n: S.i32):
    a = S.match_buffer(A, shape=(n,))
    b = S.match_buffer(B, shape=(n,))
    c = S.match_buffer(C, shape=(n,))

    for i in range(n):
        with S.block("B"):
            vi = S.axis.remap("S", [i])
            c[vi] = a[vi] + b[vi]


def test_schedule():
    sch = schedule.Schedule(add_schedule)

    block_b = sch.get_block("B")
    (i,) = sch.get_loops(block_b)
    tec, _, vl = sch.split(i, [4, None, 32])

    sch.bind_tec(tec)
    sch.vectorize(vl)

    n = 1371
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = a + b

    bm = BuildManager()
    ex = bm.build(sch.mod)

    py_out = np.empty(n, dtype)
    add_schedule(a, b, py_out, n)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, b, npu_out, n)
    assert_allclose(npu_out, gt_out)


@S.prim_func
def add_clike(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global"), n: S.i32):
    for i in range(n // 32):
        for j in S.vectorized(32):
            c[i * 32 + j] = a[i * 32 + j] + b[i * 32 + j]
    for i in S.vectorized(n % 32):
        c[n // 32 * 32 + i] = a[n // 32 * 32 + i] + b[n // 32 * 32 + i]


def test_clike():
    n = 1371
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = a + b

    bm = BuildManager()
    ex = bm.build(add_clike)

    py_out = np.empty(n, dtype)
    add_clike(a, b, py_out, n)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, b, npu_out, n)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_schedule()
    test_clike()
