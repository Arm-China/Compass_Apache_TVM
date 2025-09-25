# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def vcompt_gentype(vdtype, mask):
    @S.prim_func
    def vcompt_gentype_func(x: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vcompt(x[0], mask)

    return vcompt_gentype_func


def get_gt_output(x, mask):
    gt_out = np.where(mask, x, None)
    gt_out = filter(lambda v: v is not None, gt_out.tolist())
    gt_out = np.array(list(gt_out))
    zeros = [0] * (len(x) - len(gt_out))
    gt_out = np.concatenate([gt_out, zeros]).astype(x.dtype)
    return gt_out


@pytest.mark.parametrize(
    "dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32", "bfloat16")
)
def test_vcompt_gentype(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes

    x = rand(n, dtype)
    mask = rand(n, "bool")
    gt_out = get_gt_output(x, mask)

    f_vcompt = vcompt_gentype(vdtype, mask)
    bm = BuildManager()
    ex = bm.build(f_vcompt)

    py_out = np.empty(n, dtype)
    f_vcompt(x, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(x, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vcompt_gentype("int8")
    test_vcompt_gentype("bfloat16")
