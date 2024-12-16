# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_vcls(vdtype, mask):
    mask = mask.reshape((-1, vdtype.lanes))

    @S.prim_func
    def vcls_func(a: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vcls(a[0], mask[0])

        vout1 = S.cast(0, vdtype.with_int())
        vout1 = S.vcls(a[1], mask[1], out_sign="s")
        out[1] = vout1

        vout2 = S.cast(0, vdtype.with_uint())
        vout2 = S.vcls(a[2], mask[2], out_sign="u")
        out[2] = vout2

    return vcls_func


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32"))
def test_vcls_gentype(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes * 3
    x = rand(n, dtype)
    mask = rand(n, "bool")

    x_binary_list = [np.binary_repr(y, vdtype.bits) for y in x.tolist()]
    x_cls_list = np.array([len(s) - len(s.lstrip(s[0])) - 1 for s in x_binary_list], dtype=dtype)
    gt_out = np.where(mask, x_cls_list, 0)

    f_vcls = gen_vcls(vdtype, mask)
    bm = aipu.tir.BuildManager()
    ex = bm.build(f_vcls)

    py_out = np.empty(n, dtype)
    f_vcls(x, py_out)
    testing.assert_allclose(py_out[mask], gt_out[mask])

    aipu_out = np.empty(n, dtype)
    ex(x, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


if __name__ == "__main__":
    test_vcls_gentype("int8")
