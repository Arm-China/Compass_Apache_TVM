# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


dtype = "uint32"
n = 8 * 4


@S.prim_func
def max_reduction_1d(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global")):
    v_max = S.uint32x8(0)
    for vi in range(n // 8):
        va = S.vload(a + vi * 8)
        v_max = S.max(v_max, va)
    v_max = S.vrpmax(v_max)
    b[0] = v_max[0]


batch = 8
NUM_TEC = 4
each_tec_batch = batch // NUM_TEC


@S.prim_func
def max_reduction_2d(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global")):
    # a [batch,n]
    # b [batch]
    for ti in S.tec_range(0, NUM_TEC):
        for loop_t in range(each_tec_batch):
            off_batch = ti * each_tec_batch + loop_t
            # 1d-max reduction
            v_max = S.uint32x8(0)
            for vi in range(n // 8):
                va = S.vload(a + off_batch * n + vi * 8)
                v_max = S.max(v_max, va)
            v_max = S.vrpmax(v_max)
            # store output
            b[off_batch] = v_max[0]


def test_max_reduction_1d():
    # build the kernel
    bm = BuildManager(target="X2_1204")
    ex = bm.build(max_reduction_1d)

    # input data
    a = rand(n, dtype, low=0, high=64)

    # run on PySim
    py_out = np.zeros((1,), dtype=dtype)
    max_reduction_1d(a, py_out)

    # run on Compass simulator
    npu_out = np.zeros((1,), dtype=dtype)
    ex(a, npu_out)

    # verify result
    gt = a.max(axis=0)
    print("test_max_reduction_1d: ")
    print(f"npu_out = {npu_out[0]}")
    print(f"py_out   = {py_out[0]}")
    print(f"gt       = {gt}")
    assert py_out[0] == gt
    assert npu_out[0] == gt
    print("=============== SUCCESS ! ===============")


def test_max_reduction_2d():
    # build the kernel
    bm = BuildManager(target="X2_1204")
    ex = bm.build(max_reduction_2d)

    # input data
    a = rand(n * batch, dtype, low=0, high=100)

    # run on PySim
    py_out = np.zeros((batch,), dtype=dtype)
    max_reduction_2d(a, py_out)

    # run on Compass simulator
    npu_out = np.zeros((batch,), dtype=dtype)
    ex(a, npu_out)

    # verify result
    gt = a.reshape((batch, n)).max(axis=1)
    print("test_max_reduction_2d: ")
    print(f"npu_out = {npu_out}")
    print(f"py_out   = {py_out}")
    print(f"gt       = {gt}")

    assert_allclose(py_out, gt)
    assert_allclose(npu_out, gt)
    print("=============== SUCCESS ! ===============")


if __name__ == "__main__":
    test_max_reduction_1d()
    test_max_reduction_2d()
