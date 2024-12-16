# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_vpcnt_func(vdtype, mask):
    mask = mask.reshape((-1, vdtype.lanes))

    @S.prim_func
    def vpcnt_func(a: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vpcnt(a[0], mask[0])

        vout1 = S.cast(0, vdtype.with_int())
        vout1 = S.vpcnt(a[1], mask[1], out_sign="s")
        out[1] = vout1

        vout2 = S.cast(0, vdtype.with_uint())
        vout2 = S.vpcnt(a[2], mask[2], out_sign="u")
        out[2] = vout2

    return vpcnt_func


def count_non_zero_bit(x, bits):
    if x < 0:
        x = 2**bits + x
    binary_str = bin(x)[2:]
    n = len(binary_str)
    # '0' * (bits-n) + binary_str
    ret = 0
    for i in range(n):
        if binary_str[i] != "0":
            ret += 1
    return ret


def get_vpcnt_gt(a, vdtype, mask):
    out = [count_non_zero_bit(x, vdtype.bits) for x in a]
    out = np.array(out, dtype=vdtype.element_of)
    return np.where(mask, out, 0)


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_all_vpcnt(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes * 3
    a = rand(n, dtype)
    mask = rand(n, "bool")
    gt_out = get_vpcnt_gt(a, vdtype, mask)

    prim_func = gen_vpcnt_func(vdtype, mask)
    bm = aipu.tir.BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, py_out)
    testing.assert_allclose(py_out[mask], gt_out[mask])

    aipu_out = np.empty(n, dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out[mask], gt_out[mask])


if __name__ == "__main__":
    test_all_vpcnt("int8")
    test_all_vpcnt("uint32")
