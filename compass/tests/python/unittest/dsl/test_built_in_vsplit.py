# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vsplit_func(dtype, factor):
    @S.prim_func
    def vsplit_func(inp: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        va = S.vload(inp, lanes=24 * factor)
        va0, va1, va2 = S.vsplit(va)
        va0 += 1
        va1 += 1
        va2 += 1
        S.vstore(va0, out)
        S.vstore(va1, out + 8 * factor)
        S.vstore(va2, out + 16 * factor)

    return vsplit_func


@pytest.mark.parametrize("dtype", ("bfloat16", "int32"))
def test_vsplit(dtype):
    factor = 1 if "32" in dtype else 2
    n = 24 * factor
    a = rand(n, dtype)

    gt_out = a + 1

    py_func = gen_vsplit_func(dtype, factor)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype)
    py_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vsplit("bfloat16")
    test_vsplit("int32")
