# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import testing
from tvm.aipu.utils import rand, hw_native_vdtype
from tvm.aipu import script as S


FP16_ELEMS_ON_LSRAM = 32 * 1024 // 2
HALF_FP16_ELEMS_ON_LSRAM = FP16_ELEMS_ON_LSRAM // 2
FP16x16_ELEMS_ON_LSRAM = FP16_ELEMS_ON_LSRAM // 16
HALF_FP16x16_ELEMS_ON_LSRAM = FP16x16_ELEMS_ON_LSRAM // 2


def gen_reduce_mean():
    dtype = "float16"
    vdtype = hw_native_vdtype(dtype)

    @S.prim_func
    def reduce_h_on_hxw(in0: S.ptr(dtype, "global"), out0: S.ptr(dtype, "global"), h: S.i32, w: S.i32):
        lsram_ptr = S.alloc(FP16x16_ELEMS_ON_LSRAM, vdtype, scope="lsram")
        tec_cnt = S.get_local_size()
        tid = S.get_local_id()

        w_per_tec = S.ceildiv(w, tec_cnt)
        w_base_tec = tid * w_per_tec
        w_cur_tec = S.clip(w - w_base_tec, min_val=0, max_val=w_per_tec)

        tile_w = vdtype.lanes
        tile_h = S.min(FP16_ELEMS_ON_LSRAM // tile_w, h)

        for w_times in range(S.ceildiv(w_cur_tec, tile_w)):
            w_start = w_times * tile_w
            w_cur = S.min(w_cur_tec - w_start, tile_w)
            w_cur_v_aligned = S.ceildiv(w_cur, vdtype.lanes)
            w_cur_aligned = w_cur_v_aligned * vdtype.lanes

            vout_fp32x16 = S.vbcast(S.fp32(0), lanes=vdtype.lanes)
            for h_times in range(S.ceildiv(h, tile_h)):
                h_start = h_times * tile_h
                h_cur = S.min(h - h_start, tile_h)

                # Copy input to lsram.
                S.dma_copy(
                    dst=lsram_ptr.as_ptr(dtype),
                    src=in0 + h_start * w + w_base_tec + w_start,
                    width=w_cur,
                    src_stride=w,
                    times=h_cur,
                    dst_stride=w_cur_aligned,
                )

                for hidx in range(h_cur):
                    vout_fp32x16 += S.cast(lsram_ptr[hidx * w_cur_v_aligned], "fp32")
            vout_fp32x16 /= h
            vout_fp16x16 = S.cast(vout_fp32x16, "fp16")

            # Copy lsram to output.
            S.vstore(vout_fp16x16, out0 + w_base_tec + w_start, mask=S.tail_mask(w_cur, vdtype.lanes))

    @S.prim_func
    def reduce_w_on_hxw(in0: S.ptr(dtype, "global"), out0: S.ptr(dtype, "global"), h: S.i32, w: S.i32):
        lsram_ptr = S.alloc(FP16x16_ELEMS_ON_LSRAM, vdtype, scope="lsram")
        tec_cnt = S.get_local_size()
        tid = S.get_local_id()

        h_per_tec = S.ceildiv(h, tec_cnt)
        h_base_tec = tid * h_per_tec
        h_cur_tec = S.clip(h - h_base_tec, min_val=0, max_val=h_per_tec)

        tile_h = 1
        tile_w = S.min(w, FP16_ELEMS_ON_LSRAM // tile_h)
        tile_w = S.max(tile_w / vdtype.lanes, 1) * vdtype.lanes  # Align with hw vector lanes.

        for h_times in range(S.ceildiv(h_cur_tec, tile_h)):
            h_start = h_times * tile_h

            vout_fp32x16 = S.vbcast(S.fp32(0), lanes=vdtype.lanes)
            for w_times in range(S.ceildiv(w, tile_w)):
                w_start = w_times * tile_w
                w_cur = S.min(w - w_start, tile_w)
                w_cur_v_aligned = S.ceildiv(w_cur, vdtype.lanes)
                w_cur_aligned = w_cur_v_aligned * vdtype.lanes

                # Init tail lsram with 0.
                S.vstore(S.fp16x16(0), lsram_ptr + w_cur_v_aligned - 1)

                # Copy input to lsram.
                S.dma_copy(
                    dst=lsram_ptr.as_ptr(dtype),
                    src=in0 + (h_base_tec + h_start) * w + w_start,
                    width=w_cur,
                    dst_stride=w_cur_aligned,
                )

                for widx in range(w_cur_v_aligned):
                    vout_fp32x16 += S.cast(lsram_ptr[widx], "fp32")

            vout_fp32x8_l, vout_fp32x8_h = S.vsplit(vout_fp32x16)
            vout_fp32x8 = S.vrpadd(vout_fp32x8_l + vout_fp32x8_h) / w
            vout_fp16x16 = S.cast(vout_fp32x8, "fp16")

            # Copy lsram to output.
            S.vstore(vout_fp16x16, out0 + h_base_tec + h_start, mask="1T")

    @S.prim_func(is_entry=True)
    def mean_fp16(in0: S.ptr(dtype, "global"), out0: S.ptr(dtype, "global"), b: S.i32, h: S.i32, w: S.i32):
        if w != 1:
            n = h * w
            for batch_idx in range(b):
                reduce_h_on_hxw(in0 + batch_idx * n, out0 + batch_idx * w, h, w)
        else:
            h, w = b, h
            reduce_w_on_hxw(in0, out0, h, w)

    return mean_fp16


def norm_shape_axis(x_shape, axis):
    # Make shape be equivalent to 3D
    axis = axis + len(x_shape) if axis < 0 else axis
    assert len(x_shape) > axis
    # Make axis be equivalent to 1
    new_3d_shape = (
        int(np.multiply.reduce(x_shape[:axis])),
        x_shape[axis],
        int(np.multiply.reduce(x_shape[axis + 1 :])),
    )
    new_axis = 1
    return new_3d_shape, new_axis


def gen_x_shape_axis(reduce_type):
    x_shape = rand(3, "int32", low=1, high=100, enable_corner_values=False)
    if reduce_type == "reduce_h_on_hxw":
        return x_shape, 0  # axis equals 0
    elif reduce_type == "reduce_w_on_hxw":
        return x_shape, 2  # axis equals 2
    else:  # reduce_h_on_hxw with batch times
        return x_shape, 1  # axis equals 1


def run_test(x_shape_3d, axis, prim_func, ex):
    dtype = "float16"
    x = rand(x_shape_3d, dtype, low=-10, high=10).reshape(x_shape_3d)

    out_shape_3d = (x_shape_3d[0], 1, x_shape_3d[2])
    gt_out = np.mean(x, axis=axis, keepdims=True).astype(dtype).flatten()
    out_elems = np.product(out_shape_3d)

    py_out = np.zeros(out_elems, dtype=dtype)
    prim_func(x, py_out, x_shape_3d[0], x_shape_3d[1], x_shape_3d[2])
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.zeros(out_elems, dtype=dtype)
    ex(x, aipu_out, x_shape_3d[0], x_shape_3d[1], x_shape_3d[2])
    testing.assert_allclose(aipu_out, gt_out)


@pytest.mark.parametrize("reduce_type", ("reduce_h_on_hxw", "reduce_w_on_hxw", "batch_x_reduce_h_on_hxw"))
def test_reduce_mean_fp16(reduce_type):
    run_times = 1
    prim_func = gen_reduce_mean()
    ex = aipu.tir.BuildManager().build(prim_func)

    for run_idx in range(run_times):
        x_shape, axis = gen_x_shape_axis(reduce_type)
        x_shape_3d, axis = norm_shape_axis(x_shape, axis)

        print(f"=> type:{reduce_type} idx(from 1): {run_idx + 1}/{run_times}, x_shape_3d:{x_shape_3d}, axis:{axis}")
        run_test(x_shape_3d, axis, prim_func, ex)


if __name__ == "__main__":
    test_reduce_mean_fp16(reduce_type="reduce_h_on_hxw")
    test_reduce_mean_fp16(reduce_type="reduce_w_on_hxw")
    test_reduce_mean_fp16(reduce_type="batch_x_reduce_h_on_hxw")
