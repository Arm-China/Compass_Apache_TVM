# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def get_gt_out(func_name, a, dtype, full_mask):
    one, zero = getattr(np, dtype)(1), getattr(np, dtype)(0)
    out = np.where(getattr(np, func_name)(a), one, zero)
    return np.where(full_mask, out, False)


def gen_isinf_isnan_func(func_name, vdtype, mask):
    sdot_func = getattr(S, func_name)

    @S.prim_func
    def isinf_isnan_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global")):
        vone = S.cast(1, vdtype)
        b[0] = S.vsel(vone, 0, sdot_func(a[0], mask))

        a_scalar = a.as_ptr(vdtype.element_of) + vdtype.lanes
        b_scalar = b.as_ptr(vdtype.element_of) + vdtype.lanes
        for i in range(vdtype.lanes):
            if sdot_func(a_scalar[i]):
                b_scalar[i] = 1
            else:
                b_scalar[i] = 0

    return isinf_isnan_func


@pytest.mark.parametrize("func_name", ("isinf", "isnan"))
@pytest.mark.parametrize("dtype", ("float16", "float32"))
def test_isinf_isnan(func_name, dtype):
    vdtype = hw_native_vdtype(dtype)
    lanes = vdtype.lanes
    n = lanes * 2

    a = rand(n, dtype)
    a[:11] = np.array(("nan",) * 5 + ("-inf", "inf") * 3, dtype)
    np.random.shuffle(a)
    mask = rand(lanes, "bool")
    gt_out = get_gt_out(func_name, a, dtype, np.concatenate((mask, (True,) * lanes)))

    py_func = gen_isinf_isnan_func(func_name, vdtype, mask)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_isinf_isnan("isinf", "float32")
    test_isinf_isnan("isnan", "float16")
