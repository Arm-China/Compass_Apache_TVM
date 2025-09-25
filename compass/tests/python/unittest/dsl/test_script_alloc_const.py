# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


def vadd_const_generic(n, dtype, data):
    @S.prim_func
    def vadd_const_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global")):
        c = S.alloc_const((n,), dtype, data)
        for i in range(n):
            b[i] = a[i] + c[i]

    return vadd_const_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_add_const(dtype):
    n = 256
    a = rand((n,), dtype)
    data = list(range(n))
    gt_out = a + np.array(data, dtype)

    py_func = vadd_const_generic(n, dtype, data)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_add_const("float16")
