# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype, schedule
from tvm.compass.dsl.testing import rand, assert_allclose


dtype = "int16"
vdtype = hw_native_vdtype(dtype)
INT16_ELEMS_ON_LSRAM = 32 * 1024 // 2
INT16x16_ELEMS_ON_LSRAM = INT16_ELEMS_ON_LSRAM // 16


@S.prim_func()
def unary_vector_func(in0: S.ptr(dtype, "global"), out0: S.ptr(dtype, "global"), n: S.i32, scale: S.i16):
    lsram_ptr = S.alloc(INT16x16_ELEMS_ON_LSRAM, "int16x16", scope="lsram")
    tec_cnt = S.get_local_size()
    tid = S.get_local_id()

    elems_per_tec = S.ceildiv(n, tec_cnt)
    elems_cur_tec = S.clip(n - tid * elems_per_tec, min_val=0, max_val=elems_per_tec)

    offset_cur_tec = tid * elems_per_tec
    for lsram_idx in range(S.ceildiv(elems_cur_tec, INT16_ELEMS_ON_LSRAM)):
        elems_cur_lsram = S.min(INT16_ELEMS_ON_LSRAM, elems_cur_tec - lsram_idx * INT16_ELEMS_ON_LSRAM)
        offset_cur_lsram = offset_cur_tec + lsram_idx * INT16_ELEMS_ON_LSRAM

        S.dma_copy(lsram_ptr.as_ptr(dtype), in0 + offset_cur_lsram, elems_cur_lsram)
        for vec_idx in range(S.ceildiv(elems_cur_lsram, 16)):
            lsram_ptr[vec_idx] = lsram_ptr[vec_idx] * scale
        S.dma_copy(out0 + offset_cur_lsram, lsram_ptr.as_ptr(dtype), elems_cur_lsram)


@S.prim_func
def unary_scalar_func(A: S.ptr(dtype, "global"), B: S.ptr(dtype, "global"), size: S.int32, scale: S.i16):
    a = S.match_buffer(A, shape=(size,))
    b = S.match_buffer(B, shape=(size,))

    for i in range(size):
        with S.block("B"):
            vi = S.axis.remap("S", [i])
            b[vi] = a[vi] * scale


def test_elementwise_unary():
    n = 100000
    a = rand(n, dtype)
    scale = rand(1, dtype)
    gt_out = a * scale
    bm = BuildManager()

    sch = schedule.elementwise_unary(unary_scalar_func, "X2_1204", "B")
    ex_sch = bm.build(sch.mod)

    py_out = np.empty(n, dtype=dtype)
    unary_scalar_func(a, py_out, n, scale)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=dtype)
    ex_sch(a, npu_out, n, scale)
    assert_allclose(npu_out, gt_out)

    ex = bm.build(unary_vector_func)

    py_out = np.empty(n, dtype=dtype)
    unary_vector_func(a, py_out, n, scale)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=dtype)
    ex(a, npu_out, n, scale)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_elementwise_unary()
