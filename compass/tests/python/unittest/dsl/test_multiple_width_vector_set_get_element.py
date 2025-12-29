# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


@S.prim_func
def vector_set_get_element_func(inp: S.ptr("int32", "global"), out: S.ptr("int32", "global")):
    va0 = S.vload(inp)
    va1 = S.vload(inp + 8, lanes=16)
    va2 = S.vload(inp + 24, lanes=24)
    va3 = S.vload(inp + 48, lanes=32)
    va0[3] = 33
    va1[14] = 24
    va2[16] = 12
    va3[29] = 45
    out[0] = va0[3]
    out[1] = va1[14]
    out[2] = va2[16]
    out[3] = va3[29]
    out[4] = S.vload(inp, lanes=80)[55]


def test_vector_set_get_element():
    dtype = "int32"
    n = 80
    a = rand(n, dtype)
    gt_out = np.array([33, 24, 12, 45, 0], dtype=dtype)
    gt_out[4] = a[55]

    bm = BuildManager()
    ex = bm.build(vector_set_get_element_func)

    py_out = np.empty(5, dtype=dtype)
    vector_set_get_element_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(5, dtype=dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vector_set_get_element()
