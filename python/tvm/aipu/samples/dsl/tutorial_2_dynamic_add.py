# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.aipu import script as S, testing
from tvm import aipu
from tvm.aipu.utils import rand, hw_native_vdtype


FP16_ELEMS_ON_LSRAM = 32 * 1024 // 2 // 2
FP16x16_ELEMS_ON_LSRAM = FP16_ELEMS_ON_LSRAM // 16


def gen_add_dynamic(dtype):
    vdtype = hw_native_vdtype(dtype)

    @S.prim_func
    def add_dynamic(in0: S.ptr(dtype, "global"), in1: S.ptr(dtype, "global"), out: S.ptr(dtype, "global"), n: S.i32):
        lsram_in0 = S.alloc(FP16x16_ELEMS_ON_LSRAM, vdtype, scope="lsram")
        lsram_in1 = S.alloc(FP16x16_ELEMS_ON_LSRAM, vdtype, scope="lsram")
        tec_cnt = S.get_local_size()
        tid = S.get_local_id()

        elems_per_tec = S.ceildiv(n, tec_cnt)
        elems_cur_tec = S.clip(n - tid * elems_per_tec, min_val=0, max_val=elems_per_tec)

        offset_cur_tec = tid * elems_per_tec
        for lsram_idx in range(S.ceildiv(elems_cur_tec, FP16_ELEMS_ON_LSRAM)):
            elems_cur_lsram = S.min(FP16_ELEMS_ON_LSRAM, elems_cur_tec - lsram_idx * FP16_ELEMS_ON_LSRAM)
            offset_cur_lsram = offset_cur_tec + lsram_idx * FP16_ELEMS_ON_LSRAM

            S.dma_copy(lsram_in0.as_ptr(dtype), in0 + offset_cur_lsram, elems_cur_lsram)
            S.dma_copy(lsram_in1.as_ptr(dtype), in1 + offset_cur_lsram, elems_cur_lsram)
            for vec_idx in range(S.ceildiv(elems_cur_lsram, vdtype.lanes)):
                lsram_in0[vec_idx] = S.vadd(lsram_in0[vec_idx], lsram_in1[vec_idx])
            S.dma_copy(out + offset_cur_lsram, lsram_in0.as_ptr(dtype), elems_cur_lsram)

    return add_dynamic


def test_dynamic_add():
    dtype = "float16"
    n = 3000

    # input data
    a = rand(n, dtype, low=-100, high=100)
    b = rand(n, dtype, low=-100, high=100)
    gt_out = a + b

    # build the kernel
    py_func = gen_add_dynamic(dtype)
    bm = aipu.tir.BuildManager(target="X2_1204")
    ex = bm.build(py_func)

    # run python simulator
    py_out = np.zeros((n,), dtype=dtype)
    py_func(a, b, py_out, n)

    # run AIPU simulator
    aipu_out = np.zeros((n,), dtype=dtype)
    ex(a, b, aipu_out, n)

    # verify result
    print(f"a[:4]       ={a[:4]}")
    print(f"b[:4]       ={b[:4]}")
    print(f"aipu_out[:4]={aipu_out[:4]}")
    print(f"gt_out[:4]  ={gt_out[:4]}")

    testing.assert_allclose(py_out, gt_out)
    testing.assert_allclose(aipu_out, gt_out)
    print("=============== SUCCESS ! ===============")


if __name__ == "__main__":
    test_dynamic_add()
