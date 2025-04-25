# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import rand


@S.prim_func
def get_val(v0: S.i8, v1: S.i32 = 20, v2: S.fp16 = 30, v3: S.fp32 = 40) -> S.i32:
    return v0 + v1 + v2 + v3


@S.prim_func
def set_val(out: S.ptr("i32", "global"), v0: S.i32, v1: S.fp16 = 30):
    out[0] = v0 + v1


@S.prim_func
def func_with_default(placeholder: S.ptr("i32", "global"), out: S.ptr("i32", "global")):
    # general usage
    out[0] = get_val(v0=11, v1=22, v2=33, v3=44)
    out[1] = get_val(11, v1=22, v2=33, v3=44)
    out[2] = get_val(11, 22, v2=33, v3=44)
    out[3] = get_val(11, 22, 33, v3=44)

    # args
    out[4] = get_val(11, 22, 33, 44)
    out[5] = get_val(11, 22, 33)
    out[6] = get_val(11, 22)
    out[7] = get_val(11)

    # args + kwargs
    out[8] = get_val(11, 22, v2=33)
    out[9] = get_val(11, v1=22, v2=33)
    out[10] = get_val(11, v1=22)
    out[11] = get_val(v0=11)

    # chaos args + kwargs
    out[12] = get_val(11, v2=22)
    out[13] = get_val(v0=11, v3=22)
    out[14] = get_val(v3=11, v0=22)
    out[15] = get_val(11, v3=11, v2=22, v1=5)
    out[16] = get_val(11, v3=11, v1=5)
    out[17] = get_val(v3=44, v2=33, v1=22, v0=11)

    # complicated
    set_val(out + 18, 15)
    set_val(out + 19, v0=30, v1=40)
    set_val(out + 20, v1=30, v0=40)
    set_val(v1=30, v0=40, out=out + 21)
    set_val(v0=40, out=out + 22)


def get_gt_out(dtype):
    # general usage
    ret = [110, 110, 110, 110]
    # args
    ret += [110, 106, 103, 101]
    # args + kwargs
    ret += [106, 106, 103, 101]
    # chaos args + kwargs
    ret += [93, 83, 83, 49, 57, 110]
    # complicated
    ret += [45, 70, 70, 70, 70]
    return np.array(ret, dtype)


def test_sub_func_with_default_value():
    dtype = "int32"
    gt_out = get_gt_out(dtype)
    n = gt_out.size
    a = rand(n, dtype)

    bm = aipu.tir.BuildManager()
    ex = bm.build(func_with_default)

    py_out = np.empty(n, dtype=dtype)
    func_with_default(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_sub_func_with_default_value()
