# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand

vdtype = hw_native_vdtype("float16")
lsram_items = 32 * 1024 // vdtype.bytes
vec_items = vdtype.lanes


@S.prim_func
def exp_calc(src: S.ptr("fp16", "global"), dst: S.ptr("fp16", "global"), n: S.int32):
    lsram_buf = S.alloc_buffer((lsram_items,), "float16", scope="lsram")
    per_tec_items = (n + (4 - 1)) >> 2

    for i in S.tec_range(4):
        cur_tec_items = S.min(n - i * per_tec_items, per_tec_items)
        move_cnt = (cur_tec_items + (lsram_items - 1)) // lsram_items

        for j in range(move_cnt):
            cur_move_items = S.min(cur_tec_items - j * lsram_items, lsram_items)
            base = i * per_tec_items + j * lsram_items
            S.dma_copy(lsram_buf, src + base, cur_move_items)
            vec_cnt = (cur_move_items + vec_items - 1) // vec_items

            for k in range(vec_cnt):
                for l in S.vectorized(vec_items):
                    lsram_buf[k * vec_items + l] = S.exp(lsram_buf[k * vec_items + l])
            S.dma_copy(dst + base, lsram_buf, cur_move_items)


def test_exp_calc():
    inp_shape = rand(4, "int32", 10, 20)
    inp_arr = rand(inp_shape.tolist(), "float16", -1, 3)
    n = inp_arr.size
    gt_out = np.exp(inp_arr)

    bm = aipu.tir.BuildManager()
    ex = bm.build(exp_calc)

    py_out = np.empty(inp_shape, dtype=np.float16)
    exp_calc(inp_arr, py_out, n)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(inp_shape, dtype=np.float16)
    ex(inp_arr, aipu_out, n)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_exp_calc()
