# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.script import tir as T
from tvm.tir.schedule.testing import assert_structural_equal_ignore_global_symbol
from tvm.compass.dsl import BuildManager, script as S, schedule
from tvm.compass.dsl.testing import rand, assert_allclose


dtype = "int32"


@S.prim_func
def add(A: S.ptr(dtype, "global"), B: S.ptr(dtype, "global"), C: S.ptr(dtype, "global"), size: S.int32):
    a = S.match_buffer(A, shape=(size,))
    b = S.match_buffer(B, shape=(size,))
    c = S.match_buffer(C, shape=(size,))

    for i in range(size):
        with S.block("B"):
            vi = S.axis.remap("S", [i])
            c[vi] = a[vi] + b[vi]


@S.prim_func
def add_split(A: S.ptr(dtype, "global"), B: S.ptr(dtype, "global"), C: S.ptr(dtype, "global"), size: S.int32):
    a = S.match_buffer(A, shape=(size,))
    b = S.match_buffer(B, shape=(size,))
    c = S.match_buffer(C, shape=(size,))

    for i, j in S.grid(4, (size + 3) // 4):
        with S.block("B"):
            vi = S.axis.S(size, i * ((size + 3) // 4) + j)
            T.where(i * ((size + 3) // 4) + j < size)
            c[vi] = a[vi] + b[vi]


def test_split():
    sch = schedule.Schedule(add)
    block_b = sch.get_block("B")
    (i,) = sch.get_loops(block_b)
    sch.split(i, factors=[4, None])

    split = schedule.Schedule(add_split)

    assert_structural_equal_ignore_global_symbol(sch.mod["main"], split.mod["main"])


@S.prim_func
def add_vectorize(A: S.ptr(dtype, "global"), B: S.ptr(dtype, "global"), C: S.ptr(dtype, "global"), size: S.int32):
    a = S.match_buffer(A, shape=(size,))
    b = S.match_buffer(B, shape=(size,))
    c = S.match_buffer(C, shape=(size,))

    for i in S.grid((size + 7) // 8):
        for j in S.vectorized(8):
            with S.block("B"):
                vi = S.axis.S(size, i * 8 + j)
                T.where(i * 8 + j < size)
                c[vi] = a[vi] + b[vi]


def test_vectorize():
    sch = schedule.Schedule(add)
    block_b = sch.get_block("B")
    (i,) = sch.get_loops(block_b)
    _, vl = sch.split(i, factors=[None, 8])
    sch.vectorize(vl)

    vectorize = schedule.Schedule(add_vectorize)

    assert_structural_equal_ignore_global_symbol(sch.mod["main"], vectorize.mod["main"])


@S.prim_func
def add_bind(A: S.ptr(dtype, "global"), B: S.ptr(dtype, "global"), C: S.ptr(dtype, "global"), size: S.int32):
    a = S.match_buffer(A, shape=(size,))
    b = S.match_buffer(B, shape=(size,))
    c = S.match_buffer(C, shape=(size,))

    for i in S.tec_range(4):
        for j in range((size + 3) // 4):
            with S.block("B"):
                vi = S.axis.S(size, i * ((size + 3) // 4) + j)
                T.where(i * ((size + 3) // 4) + j < size)
                c[vi] = a[vi] + b[vi]


def test_bind():
    sch = schedule.Schedule(add)
    block_b = sch.get_block("B")
    (i,) = sch.get_loops(block_b)
    outer, _ = sch.split(i, factors=[4, None])
    sch.bind_tec(outer)

    bind = schedule.Schedule(add_bind)

    assert_structural_equal_ignore_global_symbol(sch.mod["main"], bind.mod["main"])


@S.prim_func
def add_compute_at(A: S.ptr(dtype, "global"), B: S.ptr(dtype, "global"), C: S.ptr(dtype, "global"), size: S.int32):
    a = S.match_buffer(A, shape=(size,))
    b = S.match_buffer(B, shape=(size,))
    c = S.match_buffer(C, shape=(size,))
    # with T.block("root"):
    a_lsram = T.alloc_buffer((size,), "int32", scope="lsram")
    b_lsram = T.alloc_buffer((size,), "int32", scope="lsram")
    c_lsram = T.alloc_buffer((size,), "int32", scope="lsram")
    for ax0_0, ax0_1 in S.grid(4, (size + 3) // 4):
        with S.block("a_lsram"):
            v0 = S.axis.spatial(size, ax0_0 * ((size + 3) // 4) + ax0_1)
            T.where(ax0_0 * ((size + 3) // 4) + ax0_1 < size)
            a_lsram[v0] = a[v0]
        with S.block("b_lsram"):
            v0 = S.axis.spatial(size, ax0_0 * ((size + 3) // 4) + ax0_1)
            T.where(ax0_0 * ((size + 3) // 4) + ax0_1 < size)
            b_lsram[v0] = b[v0]
        with S.block("B"):
            vi = S.axis.spatial(size, ax0_0 * ((size + 3) // 4) + ax0_1)
            T.where(ax0_0 * ((size + 3) // 4) + ax0_1 < size)
            c_lsram[vi] = a_lsram[vi] + b_lsram[vi]
        with S.block("c_lsram"):
            v0 = S.axis.spatial(size, ax0_0 * ((size + 3) // 4) + ax0_1)
            T.where(ax0_0 * ((size + 3) // 4) + ax0_1 < size)
            c[v0] = c_lsram[v0]


def test_compute_at():
    sch = schedule.Schedule(add)
    block_b = sch.get_block("B")
    block_lsram_a = sch.cache_read(block_b, 0, "lsram")
    block_lsram_b = sch.cache_read(block_b, 1, "lsram")
    block_lsram_c = sch.cache_write(block_b, 0, "lsram")

    # all compute at output block
    (i,) = sch.get_loops(block_lsram_c)
    _, loop = sch.split(i, factors=[4, None])

    # note the sequence is important
    sch.compute_at(block_b, loop)
    sch.compute_at(block_lsram_a, loop)
    sch.compute_at(block_lsram_b, loop)

    compute_at = schedule.Schedule(add_compute_at)

    assert_structural_equal_ignore_global_symbol(sch.mod["main"], compute_at.mod["main"])


@S.prim_func
def add_combine(A: S.ptr(dtype, "global"), B: S.ptr(dtype, "global"), C: S.ptr(dtype, "global"), size: S.int32):
    a = S.match_buffer(A, shape=(size,))
    b = S.match_buffer(B, shape=(size,))
    c = S.match_buffer(C, shape=(size,))
    # with T.block("root"):
    a_lsram = T.alloc_buffer((size,), "int32", scope="lsram")
    b_lsram = T.alloc_buffer((size,), "int32", scope="lsram")
    c_lsram = T.alloc_buffer((size,), "int32", scope="lsram")
    for i_0 in T.thread_binding(4, thread="threadIdx.x"):
        for i_1 in range((size + 8191) // 8192):
            for ax0 in range(2048):
                with S.block("a_lsram"):
                    v0 = S.axis.spatial(size, i_1 * 2048 + i_0 * ((size + 8191) // 8192) * 2048 + ax0)
                    T.where(i_1 * 2048 + i_0 * ((size + 8191) // 8192) * 2048 + ax0 < size)
                    T.reads(a[v0])
                    T.writes(a_lsram[v0])
                    a_lsram[v0] = a[v0]
            for ax0 in range(2048):
                with S.block("b_lsram"):
                    v0 = S.axis.spatial(size, i_1 * 2048 + i_0 * ((size + 8191) // 8192) * 2048 + ax0)
                    T.where(i_1 * 2048 + i_0 * ((size + 8191) // 8192) * 2048 + ax0 < size)
                    T.reads(b[v0])
                    T.writes(b_lsram[v0])
                    b_lsram[v0] = b[v0]
            for i_2 in range(256):
                for i_3 in T.vectorized(8):
                    with S.block("B"):
                        vi = S.axis.spatial(size, i_0 * ((size + 8191) // 8192 * 2048) + i_1 * 2048 + i_2 * 8 + i_3)
                        T.where(((i_0 * ((size + 8191) // 8192) + i_1) * 256 + i_2) * 8 + i_3 < size)
                        T.reads(a_lsram[vi], b_lsram[vi])
                        T.writes(c_lsram[vi])
                        c_lsram[vi] = a_lsram[vi] + b_lsram[vi]
            for ax0 in range(2048):
                with S.block("c_lsram"):
                    v0 = S.axis.spatial(size, i_1 * 2048 + i_0 * ((size + 8191) // 8192) * 2048 + ax0)
                    T.where(i_1 * 2048 + i_0 * ((size + 8191) // 8192) * 2048 + ax0 < size)
                    T.reads(c_lsram[v0])
                    T.writes(c[v0])
                    c[v0] = c_lsram[v0]


def test_schedule_combine():
    sch = schedule.Schedule(add)
    block_b = sch.get_block("B")

    # split loop
    (i,) = sch.get_loops(block_b)
    bind, loop, sram, vec = sch.split(i, factors=[4, None, 256, 8])  # pylint: disable=unused-variable

    # vectorize
    sch.vectorize(vec)

    # thread bind
    sch.bind_tec(bind)

    # cache & compute_at
    sch.read_at(loop, block_b, 0, "lsram")
    sch.read_at(loop, block_b, 1, "lsram")
    sch.write_at(loop, block_b, 0, "lsram")

    combine = schedule.Schedule(add_combine)
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], combine.mod["main"])

    bm = BuildManager()
    ex = bm.build(sch.mod)

    n = 65536
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = a + b

    py_out = np.empty(n, dtype)
    add(a, b, py_out, n)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, b, npu_out, n)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_split()
    test_vectorize()
    test_bind()
    test_compute_at()
    test_schedule_combine()
