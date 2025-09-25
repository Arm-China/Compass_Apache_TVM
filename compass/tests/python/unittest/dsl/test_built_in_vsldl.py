# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vsldl(vdtype, imm_z):
    @S.prim_func
    def vsldl_gentype_func(x: S.ptr(vdtype, "global"), y: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vsldl(x[0], y[0], imm_z)

    return vsldl_gentype_func


@pytest.mark.parametrize(
    "dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32", "bfloat16")
)
def test_vsldl_gentype(dtype):
    vdtype = hw_native_vdtype(dtype)
    lane = vdtype.lanes
    x = rand(lane, dtype)
    y = rand(lane, dtype)
    imm_z = rand(1, "uint32", high=lane, return_python_type=True)
    gt_out = np.concatenate([x[imm_z:], y[:imm_z]])

    f_vsldl = gen_vsldl(vdtype, imm_z)
    bm = BuildManager()
    ex = bm.build(f_vsldl)

    py_out = np.empty(lane, dtype)
    f_vsldl(x, y, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(lane, dtype)
    ex(x, y, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vsldl_gentype("int8")
    test_vsldl_gentype("float32")
    test_vsldl_gentype("bfloat16")
