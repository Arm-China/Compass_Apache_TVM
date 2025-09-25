# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, schedule
from tvm.compass.dsl.testing import rand, assert_allclose


dtype = "int32"
n = 35


@S.prim_func
def static_add(A: S.ptr(dtype, "global"), B: S.ptr(dtype, "global"), C: S.ptr(dtype, "global")):
    a = S.match_buffer(A, shape=(n,))
    b = S.match_buffer(B, shape=(n,))
    c = S.match_buffer(C, shape=(n,))

    for i in range(n):
        with S.block("C"):
            c[i] = a[i] + b[i]


def test_add_vectorize_tail():
    sch = schedule.Schedule(static_add)
    (i,) = sch.get_loops("C")
    _, vi = sch.split(i, factors=[None, 8])
    sch.vectorize(vi)

    bm = BuildManager()
    # mod = bm.lower(sch.mod)
    # mod.show()
    # print(mod.astext(False))
    ex = bm.build(sch.mod)
    # print(ex.c_code)

    a = np.array(list(range(n)), dtype=dtype)
    b = np.array(list(range(n)), dtype=dtype)

    c = np.empty(n, dtype)
    ex(a, b, c)
    assert_allclose(c, a + b)
    """
        __kernel void static_add(__global int* a, __global int* b, __global int* c) {
        for (int i_0 = 0; i_0 < 4; ++i_0) {
            vstore8((vload8(0, a + (i_0 * 8)) + vload8(0, b + (i_0 * 8))), 0, c + (i_0 * 8));
        }
        __vstore((__vload((__global int8*)(a + 32), __vmov_w(4095)) + __vload((__global int8*)(b + 32), __vmov_w(4095))), (__global int8*)(c + 32), __vmov_w(4095));
        }
    """


@S.prim_func
def c_like_dynamic_add(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global"), n: S.i32):
    tec_cnt = S.get_local_size()
    tid = S.get_local_id()

    cnt_per_tec = (n + tec_cnt - 1) // tec_cnt
    cur_tec_offset = tid * cnt_per_tec
    if cur_tec_offset >= n:  # For the situation that n < tec_cnt.
        return

    cur_tec_cnt = S.min(n - cur_tec_offset, cnt_per_tec)
    cur_idx = cur_tec_offset

    for _ in range(cur_tec_cnt // 8):
        c[cur_idx : cur_idx + 8] = a[cur_idx : cur_idx + 8] + b[cur_idx : cur_idx + 8]
        cur_idx += 8

    remain_cnt = cur_tec_cnt % 8
    if remain_cnt != 0:
        mask = S.tail_mask(remain_cnt, 8)
        va = S.vload(a + cur_idx, mask=mask)
        vb = S.vload(b + cur_idx, mask=mask)
        vc = S.vadd(va, vb, mask=mask)
        S.vstore(vc, c + cur_idx, mask=mask)


def test_c_like_dynamic_add():
    n = 32 * 4 + 5
    a = rand(n, dtype)
    b = rand(n, dtype)

    bm = BuildManager()
    ex = bm.build(c_like_dynamic_add)
    # print(ex.c_code)

    py_out = np.empty(n, dtype)
    c_like_dynamic_add(a, b, py_out, n)
    assert_allclose(py_out, a + b)

    npu_out = np.empty(n, dtype)
    ex(a, b, npu_out, n)
    assert_allclose(npu_out, a + b)


if __name__ == "__main__":
    test_add_vectorize_tail()
    test_c_like_dynamic_add()
