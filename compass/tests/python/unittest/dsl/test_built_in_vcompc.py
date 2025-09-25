# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def vcompc_gentype(vdtype, mask):
    @S.prim_func
    def vcompc_gentype_func(x: S.ptr(vdtype, "global"), y: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vcompc(x[0], y[0], mask)

    return vcompc_gentype_func


def get_gt_output(x, y, mask):
    gt_out = np.where(mask, x, None)
    gt_out = filter(lambda v: v is not None, gt_out.tolist())
    gt_out = np.array(list(gt_out))
    num_left = len(x) - len(gt_out)
    gt_out = np.concatenate([gt_out, y[:num_left]]).astype(x.dtype)
    return gt_out


@pytest.mark.parametrize(
    "dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32", "bfloat16")
)
def test_vcompc_gentype(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes

    x = rand(n, dtype)
    y = rand(n, dtype)
    mask = rand(n, "bool")
    gt_out = get_gt_output(x, y, mask)

    f_vcompc = vcompc_gentype(vdtype, mask)
    bm = BuildManager()
    ex = bm.build(f_vcompc)

    py_out = np.empty(n, dtype)
    f_vcompc(x, y, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(x, y, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vcompc_gentype("uint8")
    test_vcompc_gentype("bfloat16")
