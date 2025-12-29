# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
from tvm import runtime, te
from tvm.compass.dsl import BuildManager


def test_build_no_schedule():
    dtype = "uint8"
    n = runtime.convert(1000)
    A = te.placeholder((n,), name="A", dtype=dtype)
    B = te.placeholder((n,), name="B", dtype=dtype)
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    tir_func = te.create_prim_func([A, B, C])

    bm = BuildManager()
    ex = bm.build(tir_func, name="fadd")
    c_code = ex.c_code.strip()

    expect = """\
#include <compass/dsl.h>

__kernel void fadd(__global uchar* restrict A, __global uchar* restrict B, __global uchar* restrict C);

__kernel void fadd(__global uchar* restrict A, __global uchar* restrict B, __global uchar* restrict C) {
  for (int i0 = 0; i0 < 1000; i0 += 1) {
    C[i0] = (uchar)(A[i0] + B[i0]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}
""".strip()
    assert expect == c_code, f"\nExpect snippet:\n{expect}\n\nCompass C code:\n{c_code}\n"


if __name__ == "__main__":
    test_build_no_schedule()
