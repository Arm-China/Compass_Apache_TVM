# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


@S.prim_func
def return_dtype_func(
    a: S.ptr("i32", "global"), b: S.ptr("fp16", "global"), out0: S.ptr("i32", "global"), out1: S.ptr("fp16", "global")
):
    out0[0] = a[0] + 7
    out1[0] = b[0] * 7


def test_operator_return_dtype():
    dtype0, dtype1 = "int32", "float16"
    n, imm = 2, 7
    a = rand(n, dtype0)
    b = rand(n, dtype1)

    gt_out0 = np.array([a[0] + imm], dtype=dtype0)
    gt_out1 = np.array([b[0] * imm], dtype=dtype1)

    bm = BuildManager()
    ex = bm.build(return_dtype_func)

    # Check return dtype before binary operator result.
    expects = (
        "out0[0] = (a[0] + 7);",
        "out1[0] = (half)(b[0] * (half)0x1.cp+2f/*7.000000e+00*/);",
    )
    for expect in expects:
        assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nCompass C code:\n{ex.c_code}\n"

    py_out0, py_out1 = np.empty(n, dtype0), np.empty(n, dtype1)
    return_dtype_func(a, b, py_out0, py_out1)
    assert_allclose(py_out0[0], gt_out0[0])
    assert_allclose(py_out1[0], gt_out1[0])

    npu_out0, npu_out1 = np.empty(n, dtype0), np.empty(n, dtype1)
    ex(a, b, npu_out0, npu_out1)
    assert_allclose(npu_out0[0], gt_out0[0])
    assert_allclose(npu_out1[0], gt_out1[0])


if __name__ == "__main__":
    test_operator_return_dtype()
