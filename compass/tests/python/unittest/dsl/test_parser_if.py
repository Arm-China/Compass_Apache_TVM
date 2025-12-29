# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def vlsl_generic(vdtype, shift):
    @S.prim_func
    def vlsl_func(x: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        if shift < 0:
            out[0] = S.vsl(x[0], -shift)
        elif shift > 0:
            out[0] = S.vsl(x[0], shift)
        else:
            out[0] = x[0]

    return vlsl_func


@pytest.mark.parametrize("shift", (-3, 0, 4))
def test_vlsl(shift):
    dtype = "int8"
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes

    x = rand(n, dtype)
    if shift == 0:
        gt_out = x
    else:
        gt_out = x << shift if shift > 0 else x << -shift

    f_vlsl = vlsl_generic(vdtype, shift)
    bm = BuildManager()
    ex = bm.build(f_vlsl)

    npu_out = np.empty(n, dtype)
    ex(x, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vlsl(-3)
