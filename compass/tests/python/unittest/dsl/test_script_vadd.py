# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, schedule
from tvm.compass.dsl.testing import rand, assert_allclose


dtype = "float32"
n = 256 * 128


@S.prim_func
def static_add(A: S.ptr(dtype, "global"), B: S.ptr(dtype, "global"), C: S.ptr(dtype, "global")):
    a = S.match_buffer(A, shape=(n,))
    b = S.match_buffer(B, shape=(n,))
    c = S.match_buffer(C, shape=(n,))

    for i in range(n):
        with S.block("c"):
            c[i] = a[i] + b[i]


def do_schedule(func):
    sch = schedule.Schedule(func)
    (i,) = sch.get_loops("c")

    to, ti = sch.split(i, factors=[4, None])
    sch.bind_tec(to)

    lo, li = sch.split(ti, factors=[None, 4096])
    lsram_a = sch.cache_read("c", 0, "lsram")
    sch.compute_at(lsram_a, lo)
    lsram_b = sch.cache_read("c", 1, "lsram")
    sch.compute_at(lsram_b, lo)
    lsram_c = sch.cache_write("c", 0, "lsram")
    sch.reverse_compute_at(lsram_c, lo)

    _, vi = sch.split(li, factors=[None, 8])
    sch.vectorize(vi)
    return sch


def test_add():
    sch = do_schedule(static_add)
    bm = BuildManager()
    # mod = bm.lower(sch.mod)
    # mod.show()
    ex = bm.build(sch.mod)
    # print(ex.c_code)
    shape = (n,)
    a = rand(shape, dtype)
    b = rand(shape, dtype)
    c = np.empty(shape, dtype=dtype)
    ex(a, b, c)
    assert_allclose(c, a + b)

    # benchmark
    # ex.rpc_sess = get_rpc_session()
    # print("benchmark begin")
    # print(ex.benchmark(c, a, b, repeat=5,number=10))
    # print("benchmark end")


if __name__ == "__main__":
    test_add()
