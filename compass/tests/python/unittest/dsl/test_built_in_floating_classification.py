# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def get_gt_out(func_name, a, dtype, full_mask):
    one, zero = np.array(1, dtype), np.array(0, dtype)
    out = np.where(getattr(np, func_name)(a), one, zero)
    return np.where(full_mask, out, False)


def gen_isinf_isnan_func(func_name, vdtype, mask):
    sdot_func = getattr(S, func_name)

    @S.prim_func
    def isinf_isnan_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global")):
        vone = S.cast(1, vdtype)
        b[0] = S.vsel(vone, 0, sdot_func(a[0], mask))

        if not vdtype.is_bfloat16:
            a_scalar = a.as_ptr(vdtype.element_of) + vdtype.lanes
            b_scalar = b.as_ptr(vdtype.element_of) + vdtype.lanes
            for i in range(vdtype.lanes):
                if sdot_func(a_scalar[i]):
                    b_scalar[i] = 1
                else:
                    b_scalar[i] = 0
        else:
            b[1] = S.vsel(vone, 0, sdot_func(a[1]))

    return isinf_isnan_func


@pytest.mark.parametrize("func_name", ("isinf", "isnan", "isfinite"))
@pytest.mark.parametrize("dtype", ("float16", "float32", "bfloat16"))
def test_floating_classification(func_name, dtype):
    vdtype = hw_native_vdtype(dtype)
    lanes = vdtype.lanes
    n = lanes * 2

    a = rand(n, dtype)
    a[:11] = np.array((np.nan,) * 5 + (-np.inf, np.inf) * 3, dtype)
    np.random.shuffle(a)
    a[-1] = np.nan
    mask = rand(lanes, "bool")
    gt_out = get_gt_out(func_name, a, dtype, np.concatenate((mask, (True,) * lanes)))

    py_func = gen_isinf_isnan_func(func_name, vdtype, mask)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype)
    py_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_floating_classification("isinf", "float32")
    test_floating_classification("isnan", "float16")
    test_floating_classification("isnan", "bfloat16")
    test_floating_classification("isfinite", "float32")
