# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


DTYPE2CL_DTYPE = {
    "int8": "char",
    "uint8": "uchar",
    "int16": "short",
    "uint16": "ushort",
    "float16": "half",
}


def gen_explicit_conversion_func(dtype, min_val, max_val):
    @S.prim_func
    def explicit_conversion_func(inp: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        out[0] = S.clip(inp[0], min_val=min_val, max_val=max_val)

    return explicit_conversion_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "float16"))
def test_explicit_conversion(dtype):
    n = hw_native_vdtype(dtype).lanes
    inp = rand(n, dtype)
    min_val = 1
    max_val = 50
    gt_out = np.clip(inp[0], min_val, max_val).astype(dtype)

    prim_func = gen_explicit_conversion_func(dtype, min_val, max_val)
    bm = aipu.tir.BuildManager()
    ex = bm.build(prim_func)

    if dtype == "float16":
        expect = f"max(min(inp[0], ({DTYPE2CL_DTYPE[dtype]})5.000000e+01f), ({DTYPE2CL_DTYPE[dtype]})1.000000e+00f)"
    else:
        expect = f"max(min(inp[0], ({DTYPE2CL_DTYPE[dtype]})50), ({DTYPE2CL_DTYPE[dtype]})1)"
    assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nAIPU C code:\n{ex.c_code}\n"

    py_out = np.empty(n, dtype)
    prim_func(inp, py_out)
    testing.assert_allclose(py_out[0], gt_out)

    aipu_out = np.empty(n, dtype)
    ex(inp, aipu_out)
    testing.assert_allclose(aipu_out[0], gt_out)


if __name__ == "__main__":
    test_explicit_conversion("float16")
