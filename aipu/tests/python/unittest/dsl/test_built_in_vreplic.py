# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_vreplic(vdtype, uimm_idx, idx_scalar_type):
    @S.prim_func
    def vreplic_uimm_idx(x: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vreplic(x[0])
        out[1] = S.vreplic(x[0], uimm_idx)

    @S.prim_func
    def vreplic_var_idx(x: S.ptr(vdtype, "global"), var_idx: S.int32, out: S.ptr(vdtype, "global")):
        out[0] = S.vreplic(x[0])
        out[1] = S.vreplic(x[0], var_idx)

    return vreplic_var_idx if idx_scalar_type == "var" else vreplic_uimm_idx


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("index_scalar_type", ("uimm", "var"))
def test_vreplic(dtype, index_scalar_type):
    vdtype = hw_native_vdtype(dtype)
    lane = vdtype.lanes
    x = rand(lane, dtype)
    index_dtype = f"u{dtype}" if dtype.startswith("int") else dtype
    index_dtype = index_dtype.replace("float", "uint")
    index = rand(1, index_dtype, low=0, high=lane, return_python_type=True)
    gt_out = np.array([x[0]] * lane + [x[index]] * lane, dtype=x.dtype)

    f_vreplic = gen_vreplic(vdtype, index, index_scalar_type)
    bm = aipu.tir.BuildManager()
    ex = bm.build(f_vreplic)

    py_out = np.empty(lane * 2, dtype)
    py_args = (x, py_out) if index_scalar_type == "uimm" else (x, index, py_out)
    f_vreplic(*py_args)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(lane * 2, dtype)
    aipu_args = (x, aipu_out) if index_scalar_type == "uimm" else (x, index, aipu_out)
    ex(*aipu_args)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_vreplic("int8", index_scalar_type="uimm")
    test_vreplic("float32", index_scalar_type="var")
