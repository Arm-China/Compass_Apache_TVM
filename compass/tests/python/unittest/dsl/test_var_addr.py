# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose

dtype = "int32"
vdtype = hw_native_vdtype(dtype)


@S.prim_func
def add_2(x: S.ptr(dtype)):
    x[0] += 2


@S.prim_func
def main_func(inp: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
    a = S.i32(3)
    add_2(a.addr)

    a_ptr = a.addr
    a_ptr[0] += 1
    add_2(a_ptr)

    out[0] = a + inp[0]


def test_var_addr():
    n = vdtype.lanes
    a = rand(n, dtype)

    gt_out = a.copy()
    gt_out[0] = a[0] + 8

    bm = BuildManager()
    ex = bm.build(main_func)

    py_out = np.empty(n, dtype=dtype)
    main_func(a, py_out)
    assert_allclose(py_out[0], gt_out[0])

    npu_out = np.empty(n, dtype=dtype)
    ex(a, npu_out)
    assert_allclose(npu_out[0], gt_out[0])


@S.prim_func
def add_2_vdtype(x: S.ptr(vdtype)):
    x[0] = S.vadd(x[0], 2)


@S.prim_func
def main_func1(inp: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
    a = S.i32x8(3)
    a = S.max(a, 0)
    add_2_vdtype(a.addr)

    out[0] = inp[0] + a


def test_var_addr_0_dim_buffer():
    n = vdtype.lanes
    a = rand(n, dtype)

    gt_out = a.copy()
    gt_out[0] = a[0] + 5

    bm = BuildManager()
    ex = bm.build(main_func1)

    py_out = np.empty(n, dtype=dtype)
    main_func1(a, py_out)
    assert_allclose(py_out[0], gt_out[0])

    npu_out = np.empty(n, dtype=dtype)
    ex(a, npu_out)
    assert_allclose(npu_out[0], gt_out[0])


if __name__ == "__main__":
    test_var_addr()
    test_var_addr_0_dim_buffer()
