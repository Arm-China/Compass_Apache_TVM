# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype


def gen_vabs_func(vdtype, mask):
    @S.prim_func
    def vabs_func(inp: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.abs(inp[0], mask=mask)

    return vabs_func


@pytest.mark.parametrize("mask", ("32T", "15T1FT14F1T", "16F16T"))
def test_pretty_print_const_pred(mask):
    dtype = "int8"
    vdtype = hw_native_vdtype(dtype)

    py_func = gen_vabs_func(vdtype, mask)
    bm = BuildManager()

    stdout = str(bm.lower(py_func))

    expect = f'T.const_pred("{mask}")'
    assert expect in stdout, f"\nExpect snippet:\n{expect}\n\nStandard Output:\n{stdout}\n"


if __name__ == "__main__":
    test_pretty_print_const_pred("TFTF" * 8)
