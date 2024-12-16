# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
from tvm import aipu
from tvm.aipu import script as S


@S.prim_func
def sub_func0(
    a: S.ptr("fp16", "global"),
    b: S.ptr("fp16", "global"),
    c: S.ptr("fp16", "global"),
    d: S.ptr("fp16", "global"),
):
    e = a + 2

    if b[0] > 3.0:
        e[0] = 77.7

    if b[2] < 9.0:
        e = b + 7
        a[5] = 9
    else:
        e = e + 8
        e[2] = 66.6

    e = d + 8
    e[9] = 16.8


@S.prim_func
def main_func(a: S.ptr("fp16", "global"), b: S.ptr("fp16", "global"), c: S.ptr("fp16", "global"), n: S.i32):
    a1 = a + 2
    b = a1 + 1

    for i in range(n):
        a1 = a + 10

        if i < 10:
            a1 = b + 2
            a1[0] = 88.2
        else:
            a1[0] = 99.0

    b = c + 2
    d = b + 4
    sub_func0(a, b, c, d)


def test_sub_func():
    ir_mod = aipu.tir.BuildManager().lower(main_func)
    param_infos = aipu.tir.analysis.extract_prim_func_info(ir_mod).param_infos
    assert param_infos[0].is_output_tensor
    assert param_infos[1].is_input_tensor
    assert param_infos[2].is_output_tensor


@S.prim_func
def ptr_move(x: S.ptr("i32", "global")) -> S.ptr("i32", "global"):
    return x + 1


@S.prim_func
def return_ptr_sub_func_single_out(a: S.ptr("i32", "global"), out: S.ptr("i32", "global")):
    cur_out = ptr_move(out)
    cur_out[0] = a[0] + 1

    cur_out = ptr_move(ptr_move(out))
    cur_out[0] = a[1] + 1

    cur_out = ptr_move(cur_out)
    cur_out[0] = a[2] + 1


def test_return_ptr_sub_func_single_out():
    ir_mod = aipu.tir.BuildManager().lower(return_ptr_sub_func_single_out)
    param_infos = aipu.tir.analysis.extract_prim_func_info(ir_mod).param_infos
    assert param_infos[0].is_input_tensor
    assert param_infos[1].is_output_tensor


@S.prim_func
def ptr_select(x: S.ptr("i32", "global"), y: S.ptr("i32", "global"), cond: S.i32) -> S.ptr("i32", "global"):
    if cond == 1:
        return x + 1
    return y + 1


@S.prim_func
def return_ptr_sub_func_multi_out(
    a: S.ptr("i32", "global"),
    out1: S.ptr("i32", "global"),
    out2: S.ptr("i32", "global"),
    out3: S.ptr("i32", "global"),
    out4: S.ptr("i32", "global"),
):
    cur_out = ptr_select(out1, out2, 2)
    cur_out[0] = a[0] + 1

    cur_out = ptr_select(ptr_select(out1, out3, 1), out2, 3)
    cur_out[0] = a[2] + 1

    cur_out = ptr_select(out4, cur_out, 1)
    cur_out[0] = a[3] + 1


def test_return_ptr_sub_func_multi_out():
    ir_mod = aipu.tir.BuildManager().lower(return_ptr_sub_func_multi_out)
    param_infos = aipu.tir.analysis.extract_prim_func_info(ir_mod).param_infos
    assert param_infos[0].is_input_tensor
    assert param_infos[1].is_output_tensor
    assert param_infos[2].is_output_tensor
    assert param_infos[3].is_output_tensor
    assert param_infos[4].is_output_tensor


if __name__ == "__main__":
    test_sub_func()
    test_return_ptr_sub_func_single_out()
    test_return_ptr_sub_func_multi_out()
