# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
from tvm.compass.dsl import BuildManager, script as S


dtype = "int8"


@S.prim_func
def duplicate_rename(A: S.ptr(dtype, "global"), B: S.ptr(dtype, "global"), C: S.ptr(dtype, "global"), n: S.i32):
    a = S.match_buffer(A, shape=(n,))
    b = S.match_buffer(B, shape=(n,))
    c = S.match_buffer(C, shape=(n,))

    i = b[0]
    for very_long_loop_var_name in range(n):
        with S.block("B"):
            vi = S.axis.remap("S", [very_long_loop_var_name])
            c[vi] = a[vi] + b[vi]
    c[0] = i


def test_duplicate_rename():
    bm = BuildManager()
    ex = bm.build(duplicate_rename)

    expect = "for (int i_1 = 0; i_1 < n; i_1 += 1)"
    assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nCompass C code:\n{ex.c_code}\n"


@S.prim_func
def seq_rename(A: S.ptr(dtype, "global"), B: S.ptr(dtype, "global"), C: S.ptr(dtype, "global"), n: S.i32):
    a = S.match_buffer(A, shape=(n,))
    b = S.match_buffer(B, shape=(n,))
    c = S.match_buffer(C, shape=(n,))

    for very_long_loop_var_name in range(n // 32):
        with S.block("B"):
            vi = S.axis.remap("S", [very_long_loop_var_name])
            for j in S.vectorized(32):
                c[vi * 32 + j] = a[vi * 32 + j] + b[vi * 32 + j]
    for very_long_loop_var_name1 in range(n):
        with S.block("B"):
            vi = S.axis.remap("S", [very_long_loop_var_name1])
            c[n // 32 * 32 + vi] = a[n // 32 * 32 + vi] + b[n // 32 * 32 + vi]


def test_seq_rename():
    bm = BuildManager()
    ex = bm.build(seq_rename)

    for expect in ("for (int i = 0; i < (n >> 5); i += 1)", "for (int i_1 = 0; i_1 < n; i_1 += 1)"):
        assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nCompass C code:\n{ex.c_code}\n"


if __name__ == "__main__":
    test_duplicate_rename()
    test_seq_rename()
