# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_vmod(vdtype, mask):
    @S.prim_func
    def vmod_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        c[0] = S.vmod(a[0], b[0], mask)

    return vmod_func


def gt_mod(a, b, dtype):
    a = np.array(a, dtype="int64")
    b = np.array(b, dtype="int64")
    n = len(a)
    ret = []
    for i in range(n):
        if b[i] == 0:
            ret.append(0)
        else:
            x = abs(a[i]) % abs(b[i])
            if a[i] < 0:
                ret.append(-x)
            else:
                ret.append(x)
    return np.array(ret).astype(dtype)


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_all_vmod(dtype):
    print(dtype)
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    mask = rand(n, "bool")
    gt_out = gt_mod(a, b, dtype)

    prim_func = gen_vmod(vdtype, mask)
    bm = aipu.tir.BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, b, py_out)
    testing.assert_allclose(py_out[mask], gt_out[mask])

    aipu_out = np.empty(n, dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


if __name__ == "__main__":
    test_all_vmod("int8")
