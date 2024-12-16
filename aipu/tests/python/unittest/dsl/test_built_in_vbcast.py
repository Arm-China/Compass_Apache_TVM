# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_vbcast(vdtype, imm_x, mask):
    # pylint: disable=unused-variable
    @S.prim_func
    def vbcast_func(placeholder: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        unused = placeholder[0]
        dtype_imm_x = S.cast(imm_x, vdtype.element_of)
        out[0] = S.vbcast(dtype_imm_x, mask)

    return vbcast_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("has_mask", (True, False))
def test_vbcast(dtype, has_mask):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes

    placeholder = np.zeros(n, dtype=dtype)
    scalar_x = rand(1, dtype, return_python_type=True)
    mask = rand(n, "bool") if has_mask else [True] * n
    gt_out = np.array([scalar_x] * n, dtype=dtype)

    f_vbcast = gen_vbcast(vdtype, scalar_x, mask)
    bm = aipu.tir.BuildManager()
    ex = bm.build(f_vbcast)

    py_out = np.empty(n, dtype)
    f_vbcast(placeholder, py_out)
    testing.assert_allclose(py_out[mask], gt_out[mask])

    aipu_out = np.empty(n, dtype)
    ex(placeholder, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


if __name__ == "__main__":
    test_vbcast(dtype="int8", has_mask=True)
    test_vbcast(dtype="float32", has_mask=False)
