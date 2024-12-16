# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_vshfl(is_rotate_shift_var, vdtype, rotate_shift_imm):
    @S.prim_func
    def vshfl_func_with_var(
        x: S.ptr(vdtype, "global"),
        rotate_shift_var: S.i32,
        out: S.ptr(vdtype, "global"),
    ):
        out[0] = S.vshfl(x[0], rotate_shift_var)

    @S.prim_func
    def vshfl_func_with_imm(
        x: S.ptr(vdtype, "global"),
        placeholder: S.i32,
        out: S.ptr(vdtype, "global"),
    ):
        out[0] = S.vshfl(x[0], rotate_shift_imm)

    return vshfl_func_with_var if is_rotate_shift_var else vshfl_func_with_imm


def get_vshfl_gt_out(x, y):
    gt = np.concatenate([x, x[:y]])
    return gt[y:]


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vshfl(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    x = rand(n, dtype)
    rotate_shift = rand(1, "int32", low=0, high=n, return_python_type=True)
    gt_out = get_vshfl_gt_out(x, rotate_shift)

    for is_rotate_shift_var in (False, True):
        f_vshfl = gen_vshfl(is_rotate_shift_var, vdtype, rotate_shift)
        bm = aipu.tir.BuildManager()
        ex = bm.build(f_vshfl)

        py_out = np.empty(n, dtype)
        f_vshfl(x, rotate_shift, py_out)
        testing.assert_allclose(py_out, gt_out)

        aipu_out = np.empty(n, dtype)
        ex(x, rotate_shift, aipu_out)
        testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_vshfl("int8")
