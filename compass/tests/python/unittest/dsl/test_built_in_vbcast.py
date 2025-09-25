# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vbcast(vdtype, imm_x, mask):
    # pylint: disable=unused-variable
    @S.prim_func
    def vbcast_func(out: S.ptr(vdtype, "global")):
        dtype_imm_x = S.cast(imm_x, vdtype.element_of)
        out[0] = S.vbcast(dtype_imm_x, mask)

    return vbcast_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("has_mask", (True, False))
def test_vbcast(dtype, has_mask):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes

    scalar_x = rand(1, dtype, return_python_type=True)
    mask = rand(n, "bool") if has_mask else [True] * n
    gt_out = np.array([scalar_x] * n, dtype=dtype)

    f_vbcast = gen_vbcast(vdtype, scalar_x, mask)
    bm = BuildManager()
    ex = bm.build(f_vbcast)

    py_out = np.empty(n, dtype)
    f_vbcast(py_out)
    assert_allclose(py_out[mask], gt_out[mask])

    npu_out = np.empty(n, dtype)
    ex(npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


if __name__ == "__main__":
    test_vbcast(dtype="int8", has_mask=True)
    test_vbcast(dtype="float32", has_mask=False)
