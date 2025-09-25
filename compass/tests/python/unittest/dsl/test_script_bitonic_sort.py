# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose

num = 1 << 10


@S.prim_func
def log2(inp: S.i32) -> S.i32:
    ret = 0
    while inp > 1:
        inp = inp >> 1
        ret = ret + 1
    return ret


@S.prim_func
def sort_seq(src: S.ptr("i32", "shared"), begin: S.i32, end: S.i32, is_ascended: S.i32):
    size = end - begin
    loop_num = log2(size)
    for t in range(loop_num):
        step = size >> (t + 1)
        loop = 1 << t
        for i in range(loop):
            tec_step_num = (step + 3) // 4
            for tec_id in S.tec_range(4):
                for tec_step in range((step + 3) // 4):
                    cur_id = tec_id * tec_step_num + tec_step
                    if cur_id < step:
                        base0 = step * 2 * i + cur_id + begin
                        base1 = step * 2 * i + cur_id + begin + step
                        if is_ascended == 1 and src[base0] > src[base1] or is_ascended == 0 and src[base0] < src[base1]:
                            temp = src[base0]
                            src[base0] = src[base1]
                            src[base1] = temp
        S.barrier()


@S.prim_func
def cons_seq(src: S.ptr("i32", "shared")):
    step_num = log2(num) - 1
    for t in range(step_num):
        step = 1 << (t + 1)
        for i in range(num // step):
            if i % 2 == 0:
                sort_seq(src, i * step, (i + 1) * step, 1)
            else:
                sort_seq(src, i * step, (i + 1) * step, 0)


@S.prim_func
def bitonic_sort(src: S.ptr("i32", "global"), dst: S.ptr("i32", "global"), is_ascended: S.i32):
    shared_buf = S.alloc((num,), "i32", scope="shared")
    tid = S.get_local_id()
    step = num // 4

    S.dma_copy(shared_buf + tid * step, src + tid * step, step)
    S.barrier()

    cons_seq(shared_buf)
    sort_seq(shared_buf, 0, num, is_ascended)

    S.dma_copy(dst + tid * step, shared_buf + tid * step, step)
    S.barrier()


def test_bitonic_sort():
    src = rand(num, "int32", -50000, 50000)
    is_ascended = 1
    gt_out = src.copy()
    gt_out.sort()

    bm = BuildManager()
    ex = bm.build(bitonic_sort, name="test_sort")

    py_out = np.empty(num, "int32")
    bitonic_sort(src, py_out, is_ascended)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(num, "int32")
    ex(src, npu_out, is_ascended)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_bitonic_sort()
