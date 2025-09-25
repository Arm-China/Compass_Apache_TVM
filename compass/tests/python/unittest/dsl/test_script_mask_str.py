# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def mask_func(vdtype, mask_str):
    @S.prim_func
    def mask_string(inp0: S.ptr(vdtype, "global"), inp1: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vadd(inp0[0], inp1[0], mask=mask_str)

    return mask_string


@pytest.mark.parametrize("mask_str", ("16F16T", "FF28T2F", "4T"))
def test_script_mask_str(mask_str):
    dtype = "int8"
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)

    gt_out = np.zeros(n, dtype=dtype)
    start_idx, end_idx = {"16F16T": (16, 32), "FF28T2F": (3, 30), "4T": (0, 4)}[mask_str]
    gt_out[start_idx:end_idx] = a[start_idx:end_idx] + b[start_idx:end_idx]

    py_func = mask_func(vdtype, mask_str)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    assert_allclose(py_out[start_idx:end_idx], gt_out[start_idx:end_idx])

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out[start_idx:end_idx], gt_out[start_idx:end_idx])


if __name__ == "__main__":
    test_script_mask_str("16F16T")
    test_script_mask_str("FF28T2F")
    test_script_mask_str("4T")
