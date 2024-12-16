# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu, get_range
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def vabs_generic(vdtype, mask, saturate):
    @S.prim_func
    def vabs_func(a: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        c[0] = S.vabs(a[0], mask, saturate)

    return vabs_func


def gt_abs(x, dtype, mask, saturate):
    if saturate and not dtype.startswith("float"):
        ret = np.abs(x.astype("int64"), where=mask)
        return np.clip(ret, *get_range(dtype)).astype(dtype)
    return np.abs(x, where=mask).astype(dtype)


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32", "float16", "float32"))
@pytest.mark.parametrize("saturate", (False, True))
def test_vabs(dtype, saturate):
    if dtype.startswith("float") and saturate:
        pytest.skip("vabs with saturate=True doesn't support float input")

    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    # Sets the first value in the array to the lowest value of the data type
    # for testing saturation truncation
    a[0] = get_range(dtype)[0]
    mask = rand(n, "bool")
    mask[0] = True
    gt_out = gt_abs(a, dtype, mask, saturate)

    py_func = vabs_generic(vdtype, mask, saturate)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out[mask], gt_out[mask])

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


if __name__ == "__main__":
    test_vabs("int8", saturate=False)
    test_vabs("int32", saturate=True)
