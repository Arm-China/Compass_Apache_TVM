# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def const_mask_reassign_generic():
    @S.prim_func
    def const_mask_reassign_func(
        a: S.ptr("int8", "global"),
        b: S.ptr("int32", "global"),
        out: S.ptr("i32x8", "global"),
        flag: S.i32,
    ):
        mask = S.const_mask([True] * 4 + [False] * 4)
        if flag == 1:
            mask = S.const_mask([False] * 4 + [True] * 4)
        va = S.vload(a, lanes=8, mask=mask)
        # vb = S.vload(b, mask=mask)
        vb = S.vload(b)
        out[0] = va + vb

    return const_mask_reassign_func


def test_const_mask_reassign():
    dtype = "int32"
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, "int8")
    b = rand(n, "int32")

    gt_out = np.zeros(n, dtype=dtype)
    gt_out[4:] = a[4:] + b[4:]

    py_func = const_mask_reassign_generic()
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out, 1)
    assert_allclose(py_out[4:], gt_out[4:])

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out, 1)
    assert_allclose(npu_out[4:], gt_out[4:])


if __name__ == "__main__":
    test_const_mask_reassign()
