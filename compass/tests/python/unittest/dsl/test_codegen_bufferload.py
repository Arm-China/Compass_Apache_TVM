# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from tvm import te
from tvm.script import tir as T
from tvm.compass.dsl import BuildManager, schedule


def get_func(offset, dtype, n):
    @T.prim_func
    def func_ramp(A: T.handle, B: T.handle):
        a = T.match_buffer(A, shape=(n,), dtype=dtype)
        b = T.match_buffer(B, shape=(n,), dtype=dtype)
        b[T.Ramp(offset, 1, 8)] = a[T.Ramp(offset, 1, 8)]

    return func_ramp


@pytest.mark.parametrize("offset", [0, 3])
def test_codegen_bufferload(offset):
    dtype = "int32"
    n = 16
    bm = BuildManager()
    ex = bm.build(get_func(offset, dtype, n))
    print(ex.c_code)


def test_codegen_bufferload_unaligned():
    n = 35

    A = te.placeholder((n,), name="A")
    B = te.compute((n,), lambda i: A[i] + 3, name="B")
    tir_func = te.create_prim_func([A, B]).with_attr("global_symbol", "add")

    sch = schedule.Schedule(tir_func)
    (i,) = sch.get_loops("B")
    _, vi = sch.split(i, factors=[None, 8])
    sch.vectorize(vi)

    ex = BuildManager().build(sch.mod)
    expects = (
        "__vstore((__vload((__global float8*)(A + cse_var_1), ALL_TRUE_w) + (float8)3.00000000000000000e+00f), (__global float8*)(B + cse_var_1), ALL_TRUE_w);",
        "__vstore((__vload((__global float8*)(A + 32), __vmov_w(0x00000111)) + (float8)3.00000000000000000e+00f), (__global float8*)(B + 32), __vmov_w(0x00000111));",
    )
    for expect in expects:
        assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nCompass C code:\n{ex.c_code}\n"


if __name__ == "__main__":
    test_codegen_bufferload(offset=0)
    test_codegen_bufferload(offset=3)
    test_codegen_bufferload_unaligned()
