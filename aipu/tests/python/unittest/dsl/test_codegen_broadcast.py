# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing


def functor(dtype, n):
    vdtype = f"{dtype}x{n}"

    @S.prim_func
    def add_func(A: S.ptr(vdtype, "global"), B: S.ptr(vdtype, "global")):
        B[0] = A[0] + 1

    return add_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_codegen_broadcast(dtype):
    n = 256 // 8 // np.dtype(dtype).itemsize
    a = np.array(range(n), dtype)
    gt_out = a + 1

    func = functor(dtype, n)
    bm = aipu.tir.BuildManager()
    ex = bm.build(func)

    py_out = np.empty(n, dtype)
    func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_codegen_broadcast(dtype="int8")
