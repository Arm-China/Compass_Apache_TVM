# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def get_vrpadd_gt(a, mask, out_dtype):
    masked_a = np.where(mask, a, 0)
    out_sum = np.sum(masked_a, dtype=out_dtype)
    out = np.array([0] * len(a), dtype=out_dtype)
    out[0] = out_sum
    return out


def vrpadd_gen(in_vdtype, out_vdtype, mask):
    @S.prim_func
    def vrpadd_func(a: S.ptr(in_vdtype, "global"), b: S.ptr(out_vdtype, "global")):
        x = S.vrpadd(a[0], mask)[0]
        b[0] = S.vbcast(x)

    return vrpadd_func


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32", "uint8", "uint16", "uint32", "float16", "float32"))
def test_all_vrpadd(dtype):
    in_vdtype = hw_native_vdtype(dtype)
    out_vdtype = hw_native_vdtype("float32") if in_vdtype.is_float16 else in_vdtype
    n = in_vdtype.lanes
    a = np.array(range(n), dtype=dtype)
    mask = rand(n, "bool")
    gt_out = get_vrpadd_gt(a, mask, out_vdtype.element_of)

    prim_func = vrpadd_gen(in_vdtype, out_vdtype, mask)
    bm = aipu.tir.BuildManager()
    ex = bm.build(prim_func)
    # print(ex.c_code)

    py_out = np.empty(out_vdtype.lanes, out_vdtype.element_of)
    prim_func(a, py_out)
    testing.assert_allclose(py_out[0], gt_out[0])

    aipu_out = np.empty(out_vdtype.lanes, out_vdtype.element_of)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out[0], gt_out[0])


if __name__ == "__main__":
    test_all_vrpadd("int8")
    test_all_vrpadd("float16")
