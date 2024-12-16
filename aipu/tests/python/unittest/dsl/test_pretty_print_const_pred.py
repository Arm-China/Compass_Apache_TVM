# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from tvm import aipu
from tvm.aipu import script as S
from tvm.aipu.utils import hw_native_vdtype


def gen_vabs_func(vdtype, mask):
    @S.prim_func
    def vabs_func(inp: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vabs(inp[0], mask=mask)

    return vabs_func


@pytest.mark.parametrize("mask", ("32T", "15T1FT14F1T", "16F16T"))
def test_pretty_print_const_pred(mask):
    dtype = "int8"
    vdtype = hw_native_vdtype(dtype)

    py_func = gen_vabs_func(vdtype, mask)
    bm = aipu.tir.BuildManager()

    stdout = str(bm.lower(py_func))

    expect = f'T.const_pred("{mask}")'
    assert expect in stdout, f"\nExpect snippet:\n{expect}\n\nStandard Output:\n{stdout}\n"


if __name__ == "__main__":
    test_pretty_print_const_pred("TFTF" * 8)
