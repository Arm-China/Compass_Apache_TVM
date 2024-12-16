# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import rand


def gen_vsplit_func(dtype):
    @S.prim_func
    def vsplit_func(inp: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        va = S.vload(inp, lanes=24)
        va0, va1, va2 = S.vsplit(va)
        va0 += 1
        va1 += 1
        va2 += 1
        S.vstore(va0, out)
        S.vstore(va1, out + 8)
        S.vstore(va2, out + 16)

    return vsplit_func


def test_vsplit():
    dtype = "int32"
    n = 24
    a = rand(n, dtype)

    gt_out = a + 1

    py_func = gen_vsplit_func(dtype)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_vsplit()
