# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from tvm import aipu, te
from tvm.script import tir as T


dtype = "int32"
n = 16


def get_func(offset):
    @T.prim_func
    def func_ramp(A: T.handle, B: T.handle):
        a = T.match_buffer(A, shape=(n,), dtype=dtype)
        b = T.match_buffer(B, shape=(n,), dtype=dtype)
        b[T.Ramp(offset, 1, 8)] = a[T.Ramp(offset, 1, 8)]

    return func_ramp


@pytest.mark.parametrize("offset", [0, 3])
def test_codegen_bufferload(offset):
    bm = aipu.tir.BuildManager()
    ex = bm.build(get_func(offset))
    print(ex.c_code)


def test_codegen_bufferload_unaligned():
    n = 35

    A = te.placeholder((n,), name="A")
    B = te.compute((n,), lambda i: A[i] + 3, name="B")
    s = te.create_schedule(B.op)
    _, xi = s[B].split(B.op.axis[0], 8)
    s[B].vectorize(xi)

    ex = aipu.tir.BuildManager().build(s, [A, B], name="add")
    print(ex.c_code)
    """
    #include <aipu/tvm_aipu.h>

    __kernel void add(__global float* restrict A, __global float* restrict B) {
      for (int i_outer = 0; i_outer < 4; ++i_outer) {
        vstore8((vload8(0, A + (i_outer * 8)) + ((float8)(3.000000e+00f, 3.000000e+00f, 3.000000e+00f, 3.000000e+00f, 3.000000e+00f, 3.000000e+00f, 3.000000e+00f, 3.000000e+00f))), 0, B + (i_outer * 8));
      }
      __vstore((__vload((__global float8*)(A + 32), __vmov_w(273)) + ((float8)(3.000000e+00f, 3.000000e+00f, 3.000000e+00f, 3.000000e+00f, 3.000000e+00f, 3.000000e+00f, 3.000000e+00f, 3.000000e+00f))), (__global float8*)(B + 32), __vmov_w(273));
    }
    """


if __name__ == "__main__":
    test_codegen_bufferload(offset=0)
    test_codegen_bufferload(offset=3)
    test_codegen_bufferload_unaligned()


"""
__kernel void unknown(__global int* a, __global int* b) {
  vstore8(vload8(0, a + 0), 0, b + 0);
}

__kernel void unknown(__global int* a, __global int* b) {
  vstore8(vload8(0, a + 3), 0, b + 3);
}

before commit:

__kernel void unknown(__global int* a, __global int* b) {
  int8 v_ = int8((3)+(1*0), (3)+(1*1), (3)+(1*2), (3)+(1*3), (3)+(1*4), (3)+(1*5), (3)+(1*6), (3)+(1*7));
  vstore8((int8(a[v_.s0],a[v_.s1],a[v_.s2],a[v_.s3],a[v_.s4],a[v_.s5],a[v_.s6],a[v_.s7])), 0, b + 3);
}
"""
