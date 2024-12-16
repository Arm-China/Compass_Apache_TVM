# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_vall_vany_func(func_name, vdtype):
    sdot_func = {"vall": S.vall, "vany": S.vany}[func_name]

    @S.prim_func
    def func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), out: S.ptr(vdtype.element_of, "global")):
        mask = a[0] > b[0]
        if sdot_func(mask):
            out[0] = 0
        else:
            out[0] = 1

    return func


@pytest.mark.parametrize("func_name", ("vall", "vany"))
@pytest.mark.parametrize("dtype", ("int8", "int16", "int32"))
def test_vall_vany(func_name, dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    va = rand(n, dtype)
    vb = rand(n, dtype)
    gt_out = np.array([0 if getattr(np, func_name[1:])(va > vb) else 1]).astype(dtype)

    prim_func = gen_vall_vany_func(func_name, vdtype)
    bm = aipu.tir.BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(1, dtype)
    prim_func(va, vb, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(1, dtype)
    ex(va, vb, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_vall_vany("vall", "int16")
    test_vall_vany("vany", "int8")
