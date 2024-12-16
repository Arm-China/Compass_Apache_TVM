# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import rand


num = 1 << 10


@S.macro
def add_one(a, b, idx):
    b[idx] = a[idx] + 1


@S.prim_func
def macro_prim_func(src: S.ptr("i32", "global"), dst: S.ptr("i32", "global")):
    shared_buf = S.alloc_buffer(num, "int32", scope="shared")
    step = num // 4

    for i in S.tec_range(4):
        S.dma_copy(shared_buf.addr_of(i * step), src + i * step, step)
    S.barrier()

    for i in S.tec_range(4):
        base = i * step
        for idx in range(step):
            add_one(src, shared_buf, idx + base)

    for i in S.tec_range(4):
        S.dma_copy(dst + i * step, shared_buf.addr_of(i * step), step)


def test_macro():
    src = rand(num, "int32", low=-50000, high=50000)
    gt_out = src + 1

    bm = aipu.tir.BuildManager()
    ex = bm.build(macro_prim_func, name="test_macro")

    aipu_out = np.empty(num, "int32")
    ex(src, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_macro()
