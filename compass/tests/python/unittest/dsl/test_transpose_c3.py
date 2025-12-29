# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_transpose_chw_hwc_c3(dtype="int8", LSRAM_SIZE=32 * 1024):
    """
    Plugin should check:
    - c == 3
    - [nchw->nhwc] : perm = [0 2 3 1]
    """
    ITEM_SIZE = np.dtype(dtype).itemsize
    LEN_V = 32 // ITEM_SIZE
    LSRAM_SIZE = LSRAM_SIZE // ITEM_SIZE
    HALF_LSRAM = LSRAM_SIZE // 2
    LSRAM_hw = HALF_LSRAM // 3 // LEN_V * LEN_V

    @S.prim_func
    def transpose_chw_hwc_c3(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), n: S.int32, c: S.int32, hw: S.int32):
        lsram = S.alloc_buffer([LSRAM_SIZE], dtype, scope="lsram")

        loop_t = S.get_local_size()
        len_t = hw // loop_t
        chw = c * hw
        for ni in range(n):
            off_n = ni * chw
            for ti in S.tec_range(loop_t):
                mov_t = S.min(hw - ti * len_t, len_t)
                off_t = ti * len_t
                len_l = S.min(mov_t, LSRAM_hw)
                loop_l = (mov_t + len_l - 1) // len_l
                for li in range(loop_l):
                    mov_l = S.min(len_l, mov_t - li * len_l)
                    off_l = off_t + li * len_l
                    S.dma_copy(lsram, a + off_n + off_l, mov_l)
                    S.dma_copy(lsram.addr_of(LSRAM_hw), a + off_l + hw, mov_l)
                    S.dma_copy(lsram.addr_of(LSRAM_hw * 2), a + off_l + 2 * hw, mov_l)
                    loop_v = (mov_l + LEN_V - 1) // LEN_V
                    for vi in range(loop_v):
                        off_v = vi * LEN_V
                        v0 = S.vload(lsram.addr_of(off_v))
                        v1 = S.vload(lsram.addr_of(off_v + LSRAM_hw))
                        v2 = S.vload(lsram.addr_of(off_v + 2 * LSRAM_hw))
                        S.vstore(v0, lsram.addr_of(HALF_LSRAM + off_v * 3), stride=3)
                        S.vstore(v1, lsram.addr_of(HALF_LSRAM + off_v * 3 + 1), stride=3)
                        S.vstore(v2, lsram.addr_of(HALF_LSRAM + off_v * 3 + 2), stride=3)
                    S.dma_copy(b + off_n + off_l * 3, lsram.addr_of(HALF_LSRAM), mov_l * 3)

    return transpose_chw_hwc_c3


def test_transpose_c3():
    n = 1
    c = 3
    h, w = 1, 32
    hw = h * w
    N = n * c * hw
    dtype = "int8"

    x = rand(N, dtype)
    gt = np.transpose(x.reshape((n, c, h, w)), (0, 2, 3, 1)).flatten()
    func = gen_transpose_chw_hwc_c3(dtype)

    bm = BuildManager()
    ex = bm.build(func)

    py_out = np.empty(N, dtype=dtype)
    func(x, py_out, n, c, hw)
    assert_allclose(py_out, gt)

    npu_out = np.empty(N, dtype=dtype)
    ex(x, npu_out, n, c, hw)
    assert_allclose(npu_out, gt)


if __name__ == "__main__":
    test_transpose_c3()
