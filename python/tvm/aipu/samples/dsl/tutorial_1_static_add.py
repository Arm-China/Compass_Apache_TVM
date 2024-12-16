# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
from tvm.aipu import script as S, testing
from tvm import aipu
from tvm.aipu.utils import rand
import numpy as np


dtype = "float16"
n = 2048


@S.prim_func
def add_static(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
    # allocate lsram with size 512
    # first half of lsram for input a: lsram_a with size 256
    # second half of lsram for input b: lsram_b with size 256
    # reuse lsram_a for lsram_c
    lsram = S.alloc_buffer([512], dtype, scope="lsram")
    lsram_a = lsram.addr_of(0)
    lsram_b = lsram.addr_of(256)

    # TEC
    # ==================================================
    # n = 2048, NUM_TEC = 4
    # each tec compute n/TEC_NUM = 2048/4 = 512 elements
    # offset of each tec is: ti* 512
    TEC_NUM = 4
    for ti in S.tec_range(TEC_NUM):
        len_t = 512
        off_t = ti * len_t

        # LSRAM
        # =====================================================
        # for each tec, compute 512 element of input a
        # the lsram size of a (lsram_a) is 256
        # the loop_num to put input data into lsram is 512/256=2
        for li in range(2):
            off_l = off_t + li * 256

            # DMA
            # ==================================================
            # move input_a from DDR to lsram_a, size = 256
            # move input_b from DDR to lsram_b, size = 256
            S.dma_copy(lsram_a, a + off_l, 256)
            S.dma_copy(lsram_b, b + off_l, 256)

            # vectorized
            # ==================================================
            # dtype = float16, vector_lane = 16
            # we have 256 elements on lsram_a
            # each vector compute 16 elements
            # the loop_num = 256/16 = 16
            for vi in range(16):
                # vload a vector float16x16 from lsram_a
                # vload a vector float16x16 from lsram_b
                va = S.vload(lsram_a + vi * 16)
                vb = S.vload(lsram_b + vi * 16)

                # vector addition
                vc = S.vadd(va, vb)  # can also use: vc = va + vb

                # store the vector into lsram
                # we reuse lsram_a for output c
                S.vstore(vc, lsram_a + vi * 16)

            # move output from LSRAM to DDR
            S.dma_copy(c + off_l, lsram_a, 256)


def test_static_add():
    # build the kernel
    bm = aipu.tir.BuildManager(target="X2_1204")
    ex = bm.build(add_static)

    # input data
    a = rand(n, dtype, low=-100, high=100)
    b = rand(n, dtype, low=-100, high=100)
    gt_out = a + b

    # run python simulator
    py_out = np.zeros((n,), dtype=dtype)
    add_static(a, b, py_out)

    # run AIPU simulator
    aipu_out = np.zeros((n,), dtype=dtype)
    ex(a, b, aipu_out)

    # verify result
    print(f"a[:4]       ={a[:4]}")
    print(f"b[:4]       ={b[:4]}")
    print(f"aipu_out[:4]={aipu_out[:4]}")
    print(f"gt_out[:4]  ={gt_out[:4]}")

    testing.assert_allclose(py_out, gt_out, atol=1e-4)
    testing.assert_allclose(aipu_out, gt_out, atol=1e-4)
    print("=============== SUCCESS ! ===============")


if __name__ == "__main__":
    test_static_add()
