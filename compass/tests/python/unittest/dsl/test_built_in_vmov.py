# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_tail_mask(vdtype, tail_imm=None):
    @S.prim_func
    def tail_var_mask(a: S.ptr(vdtype, "global"), tail: S.uint32, b: S.ptr(vdtype, "global")):
        mask = S.tail_mask(tail, vdtype.lanes)
        b[0] = S.vadd(a[0], 1, mask, r=a[0])

    @S.prim_func
    def tail_imm_mask(a: S.ptr(vdtype, "global"), placeholder: S.uint32, b: S.ptr(vdtype, "global")):
        mask = S.tail_mask(tail_imm, vdtype.lanes)
        b[0] = S.vadd(a[0], 1, mask, r=a[0])

    return tail_imm_mask if tail_imm is not None else tail_var_mask


def get_gt_out(a, n, tail, r):
    gt_out = r.copy()
    for i in range(tail):
        gt_out[i] = a[i] + 1
    return gt_out


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32"))
def test_tail_mask(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = np.array(list(range(n)), dtype=dtype)

    for is_tail_imm in (True, False):
        for tail in (0, n, rand(1, "int32", low=1, high=n, return_python_type=True)):
            gt_out = get_gt_out(a, n, tail, r=a)

            py_func = gen_tail_mask(vdtype, tail if is_tail_imm else None)
            bm = BuildManager()
            ex = bm.build(py_func)

            py_out = np.empty(n, dtype)
            py_func(a, tail, py_out)
            assert_allclose(py_out, gt_out)

            npu_out = np.empty(n, dtype)
            ex(a, tail, npu_out)
            assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_tail_mask("int32")
