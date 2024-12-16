# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_vinsert(vdtype, imm_val, value_type, imm_idx, index_type):
    @S.prim_func
    def vinsert_imm_val_imm_idx(x: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        vx = x[0]
        vx[imm_idx] = imm_val
        out[0] = vx

    @S.prim_func
    def vinsert_var_val_var_idx(x: S.ptr(vdtype, "global"), var_idx: S.i32, out: S.ptr(vdtype, "global")):
        var_val = S.cast(imm_val, vdtype.element_of)
        vx = x[0]
        vx[var_idx] = var_val
        out[0] = vx

    @S.prim_func
    def vinsert_imm_val_var_idx(x: S.ptr(vdtype, "global"), var_idx: S.i32, out: S.ptr(vdtype, "global")):
        vx = x[0]
        vx[var_idx] = imm_val
        out[0] = vx

    @S.prim_func
    def vinsert_var_val_imm_idx(x: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        var_val = S.cast(imm_val, vdtype.element_of)
        vx = x[0]
        vx[imm_idx] = var_val
        out[0] = vx

    if value_type == index_type == "imm":
        return vinsert_imm_val_imm_idx
    elif value_type == index_type == "var":
        return vinsert_var_val_var_idx
    elif value_type == "imm" and index_type == "var":
        return vinsert_imm_val_var_idx
    return vinsert_var_val_imm_idx


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("value_type", ("imm", "var"))  # scalar
@pytest.mark.parametrize("index_type", ("imm", "var"))  # scalar
def test_vinsert(dtype, value_type, index_type):
    vdtype = hw_native_vdtype(dtype)
    lane = vdtype.lanes
    x = rand(lane, dtype)
    value = rand(1, dtype, return_python_type=True)
    index = rand(1, "int32", low=0, high=lane, return_python_type=True)
    gt_out = x.copy()
    gt_out[index] = value

    f_vinsert = gen_vinsert(vdtype, value, value_type, index, index_type)
    bm = aipu.tir.BuildManager()
    ex = bm.build(f_vinsert)

    py_out = np.empty(lane, dtype)
    py_args = (x, py_out) if index_type == "imm" else (x, index, py_out)
    f_vinsert(*py_args)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(lane, dtype)
    aipu_args = (x, aipu_out) if index_type == "imm" else (x, index, aipu_out)
    ex(*aipu_args)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_vinsert("int8", value_type="imm", index_type="imm")
    test_vinsert("int16", value_type="imm", index_type="var")
    test_vinsert("float16", value_type="var", index_type="var")
    test_vinsert("float32", value_type="var", index_type="var")
