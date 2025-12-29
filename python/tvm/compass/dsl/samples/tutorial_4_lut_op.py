# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
# pylint: disable=redefined-outer-name
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


dtype = "float16"
vdtype = hw_native_vdtype(dtype)
lut_len = 512
lut_edge = 10

FP16_ELEMS_ON_LSRAM = 32 * 1024 // 2
FP16x16_ELEMS_ON_LSRAM = FP16_ELEMS_ON_LSRAM // 16


def gen_silu_lut(lut_len, lut_edge, dtype):
    # Key points of FP16 LUT generation:
    # 1. lut_x[-lut_edge, lut_edge]
    # 2. dytpe = "float16"
    # 3. pad the lut_table with the last element (lut_edge)
    x = np.linspace(-lut_edge, lut_edge, lut_len - 1)
    lut = np.zeros((lut_len,), dtype=dtype)
    for i in range(lut_len - 1):
        lut[i] = x[i] / (1 + np.exp(-x[i]))
    lut[-1] = lut_edge
    lut[0] = 0
    return lut


@S.prim_func
def compute(
    x: S.fp16x16,
    lut: S.ptr(dtype, "constant"),
    lut_inverse_delta: S.fp32,
    lut_len: S.i32,
    lut_edge: S.i32,
) -> S.fp16x16:
    # Original formula: silu(x) = x / (1 + e^(-x))
    # Here use lookup table implement with interpolation instead
    x_clipped = S.clip(x, min_val=S.fp16(-lut_edge), max_val=S.fp16(lut_edge))
    x_fp32 = S.cast(x_clipped, "fp32")
    x_idx = (x_fp32 + lut_edge) * lut_inverse_delta
    x_idxr = S.rint(x_idx - 0.5)
    x_idx_u16 = S.cast(x_idxr, "u16")
    mask_idx_ge_lutlen_m2 = x_idx_u16 >= S.u16x16(lut_len - 2)

    y = S.vsel(x, 0, mask_idx_ge_lutlen_m2)
    lut_x_idx = S.vload_gather(lut, x_idx_u16)
    lut_x_idx_plus1 = S.vload_gather(lut, x_idx_u16 + 1)
    x_idx_diff = x_idx - x_idxr
    lut_diff = S.cast(lut_x_idx_plus1 - lut_x_idx, "fp32")
    yy = S.cast(lut_diff * x_idx_diff, dtype)
    mask_xor = S.vxor(x_idx_u16 >= 0, mask_idx_ge_lutlen_m2)
    return S.vadd(yy, lut_x_idx, mask=mask_xor, r=y)


@S.prim_func
def silu_fp16(
    in0: S.ptr(dtype, "global"),
    out0: S.ptr(dtype, "global"),
    n: S.i32,
):
    lut = S.alloc_const((lut_len,), dtype, gen_silu_lut(lut_len, lut_edge, dtype))
    lut_inverse_delta = S.fp32(lut_len - 2) / (2 * lut_edge)

    lsram_ptr = S.alloc(FP16x16_ELEMS_ON_LSRAM, vdtype, scope="lsram")
    tec_cnt = S.get_local_size()
    tid = S.get_local_id()

    elems_per_tec = S.ceildiv(n, tec_cnt)
    elems_cur_tec = S.clip(n - tid * elems_per_tec, min_val=0, max_val=elems_per_tec)

    offset_cur_tec = tid * elems_per_tec
    for lsram_idx in range(S.ceildiv(elems_cur_tec, FP16_ELEMS_ON_LSRAM)):
        elems_cur_lsram = S.min(FP16_ELEMS_ON_LSRAM, elems_cur_tec - lsram_idx * FP16_ELEMS_ON_LSRAM)
        offset_cur_lsram = offset_cur_tec + lsram_idx * FP16_ELEMS_ON_LSRAM

        S.dma_copy(lsram_ptr.as_ptr(dtype), in0 + offset_cur_lsram, elems_cur_lsram)
        for vec_idx in range(S.ceildiv(elems_cur_lsram, vdtype.lanes)):
            lsram_ptr[vec_idx] = compute(lsram_ptr[vec_idx], lut, lut_inverse_delta, lut_len, lut_edge)
        S.dma_copy(out0 + offset_cur_lsram, lsram_ptr.as_ptr(dtype), elems_cur_lsram)


def get_silu_gt(x):
    n = len(x)
    inp = x.astype("float64")
    ret = np.zeros((n,), dtype="float64")
    for i in range(n):
        ret[i] = inp[i] / (1 + np.exp(-inp[i]))
    return ret.astype(x.dtype)


def test_silu():
    # build the kernel
    bm = BuildManager(target="X2_1204")
    ex = bm.build(silu_fp16)

    n = 1000
    # input data
    a = rand(n, dtype, low=-lut_edge, high=lut_edge)
    # run on PySim
    py_out = np.zeros((n,), dtype=dtype)
    silu_fp16(a, py_out, n)

    # run on Compass simulator
    npu_out = np.zeros((n,), dtype=dtype)
    ex(a, npu_out, n)

    # verify result
    print(f"a[:4]       ={a[:4]}")
    print(f"npu_out[:4]={npu_out[:4]}")
    print(f"gt_out[:4]  ={get_silu_gt(a)[:4]}")

    assert_allclose(npu_out, get_silu_gt(a), atol=1e-3)
    assert_allclose(py_out, get_silu_gt(a), atol=1e-3)
    print("=============== SUCCESS ! ===============")


if __name__ == "__main__":
    test_silu()
