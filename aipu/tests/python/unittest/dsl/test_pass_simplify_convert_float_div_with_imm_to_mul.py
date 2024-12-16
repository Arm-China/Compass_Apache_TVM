# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm import aipu
from tvm.aipu import script as S
from tvm.aipu.utils import hw_native_vdtype


@S.prim_func
def fp32_div_imm_func(a: S.ptr("fp32", "global"), b: S.ptr("fp32", "global")):
    b[0] = a[0] / 10


def test_fp32_div_imm_func():
    dtype = "float32"
    n = hw_native_vdtype(dtype).lanes
    a = np.array([np.uint32(0x42888000).view(dtype)] * n)
    gt_out = "0x40da6666"
    get_1st_view_u32 = lambda arr: hex(arr[0].view("uint32"))

    bm = aipu.tir.BuildManager()
    ex = bm.build(fp32_div_imm_func)

    expect = "b[0] = (float)(a[0] / 1.00000000000000000e+01f);"
    assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nAIPU C code:\n{ex.c_code}\n"

    py_out = np.empty(n, dtype=dtype)
    fp32_div_imm_func(a, py_out)
    msg = f'Expect py_out[0].view("uint32"): {gt_out}, but got: {get_1st_view_u32(py_out)}'
    assert get_1st_view_u32(py_out) == gt_out, msg

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, aipu_out)
    msg = f'Expect aipu_out[0].view("uint32"): {gt_out}, but got: {get_1st_view_u32(aipu_out)}'
    assert get_1st_view_u32(aipu_out) == gt_out, msg


if __name__ == "__main__":
    test_fp32_div_imm_func()
