# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import rand


@S.prim_func
def literal_int_func(a: S.ptr("u8x32", "global"), out: S.ptr("i32x8", "global")):
    u8_out = out.as_ptr("u8x32")
    u8_out[0] = a[0] + 5
    u8_out[1] = 5 + a[1]

    out[2:6] = a[2] + 300
    out[2:6] = out[2:6]  # Test for PySim, load multiple vectors as 1D data.
    out[6:10] = 300 + a[3]

    cur_i32x32_out = (out + 10).as_ptr("i32x32")
    cur_i32x32_out[0] = a[4] + S.i32(300)
    cur_i32x32_out += 1
    i32x8_out = cur_i32x32_out.as_ptr("i32x8")
    i32x8_out[:4] = S.i32(300) + a[5]


def get_literal_int_gt(a, out_n):
    ret = np.empty(out_n, "int32")
    u8_ret = ret.view("uint8")
    u8_ret[:64] = a[:64] + 5
    ret[16:] = a[64:] + 300
    return ret


def test_literal_int():
    a = rand(6 * 32, "uint8")
    out_n = 18 * 8
    gt_out = get_literal_int_gt(a, out_n)

    bm = aipu.tir.BuildManager()
    ex = bm.build(literal_int_func)

    py_out = np.empty(out_n, "int32")
    literal_int_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(out_n, "int32")
    ex.run(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


@S.prim_func
def literal_float_func(
    a: S.ptr("u8x32", "global"),
    b: S.ptr("fp16x16", "global"),
    out0: S.ptr("fp32", "global"),
    out1: S.ptr("fp16", "global"),
):
    out0[:32] = a[0] + 5.3
    out0[32:64] = 5.3 + a[1]
    out0[64:96] = a[2] + S.fp32(5.3)
    out0[96:128] = S.fp32(5.3) + a[3]

    # For float, can't know whether "5.3" can be represented by float16 or not, so can't do anything.
    out0[128:144] = b[0] + 5.3
    out0[144:160] = 5.3 + b[1]
    out0[160:176] = b[2] + S.fp32(5.3)
    out0[176:192] = S.fp32(5.3) + b[3]

    out1[:16] = b[4] + 300
    out1[16:32] = 300 + b[5]
    out1[32:48] = b[6] + S.i32(300)
    out1[48:64] = S.i32(300) + b[7]


def get_literal_float_gt(a, b, out0_n):
    gt_out0 = np.empty(out0_n, "float32")
    gt_out0[:128] = a.astype("float32") + 5.3
    gt_out0[128:] = b[: 4 * 16].astype("float32") + 5.3
    return gt_out0, b[4 * 16 :] + np.float16(300)


def test_literal_float():
    a = rand(4 * 32, "uint8")
    b = rand(8 * 16, "float16")
    out0_n = 4 * 32 + 4 * 16
    out1_n = 4 * 16
    gt_out0, gt_out1 = get_literal_float_gt(a, b, out0_n)

    bm = aipu.tir.BuildManager()
    ex = bm.build(literal_float_func)

    py_out0 = np.empty(out0_n, "float32")
    py_out1 = np.empty(out1_n, "float16")
    literal_float_func(a, b, py_out0, py_out1)
    testing.assert_allclose(py_out0, gt_out0)
    testing.assert_allclose(py_out1, gt_out1)

    aipu_out0 = np.empty(out0_n, "float32")
    aipu_out1 = np.empty(out1_n, "float16")
    ex.run(a, b, aipu_out0, aipu_out1)
    testing.assert_allclose(aipu_out0, gt_out0)
    testing.assert_allclose(aipu_out1, gt_out1)


if __name__ == "__main__":
    test_literal_int()
    test_literal_float()
