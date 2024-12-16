# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm import aipu
from tvm.aipu.utils import rand
from tvm.aipu import script as S, testing


@S.prim_func
def tuple_assign_func(a: S.ptr("int8", "global"), b: S.ptr("int8", "global"), z: S.i8):
    x, y = a[0], a[1]
    x, y = y, x
    x, y, z = z, z + x, z + y - x

    b[0] = x
    b[1] = y
    b[2] = z


def test_tuple_assign():
    dtype = "int8"
    bm = aipu.tir.BuildManager()
    ex = bm.build(tuple_assign_func)
    a = rand(2, dtype)
    z = rand(1, dtype)
    gt_out = np.array([z, a[1] + z, a[0] + z - a[1]], dtype=dtype)

    py_out = np.empty(3, dtype=dtype)
    tuple_assign_func(a, py_out, z)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(3, dtype=dtype)
    ex(a, aipu_out, z)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_tuple_assign()
