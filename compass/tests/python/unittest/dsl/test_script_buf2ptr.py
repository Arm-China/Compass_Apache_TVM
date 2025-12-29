# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_buf2ptr_func(dtype, shape):
    @S.prim_func
    def add(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        va = S.vload(a)
        vb = S.vload(b)
        vc = S.vload(c)
        out = va + vb + vc
        S.barrier()
        S.vstore(out, c)

    @S.prim_func
    def buf2ptr_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        B = S.match_buffer(b, shape)
        C = S.match_buffer(c, shape)

        va = S.vload(a)
        vb = S.vload(B)
        vc = va + vb
        S.vstore(vc, C)

        add(a, B, C.addr_of(0))

    return buf2ptr_func


def test_buf2ptr():
    shape = (8,)
    dtype = "float32"
    a = rand(shape, dtype)
    b = rand(shape, dtype)
    gt_out = a + b + (a + b)

    buf2ptr_func = gen_buf2ptr_func(dtype, shape)
    ex = BuildManager().build(buf2ptr_func)

    py_out = np.empty(shape, dtype)
    buf2ptr_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(shape, dtype)
    ex.run(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_buf2ptr()
