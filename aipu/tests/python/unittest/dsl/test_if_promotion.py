# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
from tvm import aipu, tir
from tvm.aipu import script as S


@S.prim_func
def func1(a: S.ptr("fp32", "global")):
    for tx in S.tec_range(4):
        for i in range(10):
            if tx < 3:
                a[tx * 8 + i] = a[tx * 8 + i] + 2


@S.prim_func
def func2(a: S.ptr("fp32", "global")):
    for tx in S.tec_range(4):
        for i in S.vectorized(8):
            if tx < 3:
                a[tx * 8 + i] = a[tx * 8 + i] + 2


def test_if_promotion_with_threadIdx():
    bm = aipu.tir.BuildManager()
    mod = bm.lower(func1)
    assert isinstance(mod["func1"].body.body, tir.IfThenElse) is True
    assert isinstance(mod["func1"].body.body.then_case, tir.For) is True
    ex = bm.build(func1)
    print(ex.c_code)
    """
    __kernel void func(__global float* a) {
      if (((int)get_local_id(0)) < 3) {
        for (int i = 0; i < 10; ++i) {
            a[((((int)get_local_id(0)) * 8) + i)] = (a[((((int)get_local_id(0)) * 8) + i)] + 2.000000e+00f);
        }
      }
    }
    """


def test_if_promotion_with_vectorized():
    bm = aipu.tir.BuildManager()
    mod = bm.lower(func2)
    assert isinstance(mod["func2"].body.body, tir.IfThenElse) is True
    assert isinstance(mod["func2"].body.body.then_case, tir.BufferStore) is True
    ex = bm.build(func2)
    print(ex.c_code)
    """
    __kernel void func2(__global float* a) {
    if (((int)get_local_id(0)) < 3) {
        vstore8((vload8(0, a + (((int)get_local_id(0)) * 8)) + ((float8)2.000000e+00f)), 0, a + (((int)get_local_id(0)) * 8));
    }
    }
    """


if __name__ == "__main__":
    test_if_promotion_with_threadIdx()
    test_if_promotion_with_vectorized()
