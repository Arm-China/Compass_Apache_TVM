# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


@S.prim_func
def param_pointer(a: S.ptr("fp16x32", "global"), b: S.ptr("fp16x64", "global"), c: S.ptr("fp16x16", "global")):
    new_a = a.as_ptr("fp16x16")
    new_b = b.as_ptr("fp16x16")
    for i in range(4):
        c[i] = new_a[i] + new_b[i]


def test_function_param_pointer():
    dtype = "float16"
    shape = (64,)
    a = rand(shape, dtype)
    b = rand(shape, dtype)
    gt_out = a + b

    bm = BuildManager()
    ex = bm.build(param_pointer)

    py_out = np.empty(shape, dtype=dtype)
    param_pointer(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(shape, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_function_param_pointer()
