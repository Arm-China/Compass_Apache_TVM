# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import torch
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_dma_upsample(direction, dtype, elems_inp, h_scale, w_scale, c, w, src_c_stride, dst_c_stride, dst_w_stride):
    src_scope, dst_scope = direction.split("_to_")
    sram_type = dst_scope if src_scope == "ddr" else src_scope
    num_elem_upsampled = h_scale * dst_w_stride * dst_c_stride

    @S.prim_func
    def dma_upsample_sram2ddr(inp: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        sram = S.alloc(elems_inp, dtype, scope=sram_type)
        S.dma_copy(sram, inp, elems_inp)
        S.dma_upsample(out, sram, h_scale, w_scale, c, w, src_c_stride, dst_c_stride, dst_w_stride)

    @S.prim_func
    def dma_upsample_ddr2sram(inp: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        sram = S.alloc(num_elem_upsampled, dtype, scope=sram_type)
        S.dma_upsample(sram, inp, h_scale, w_scale, c, w, src_c_stride, dst_c_stride, dst_w_stride)
        S.dma_copy(out, sram, num_elem_upsampled)

    return dma_upsample_ddr2sram if src_scope == "ddr" else dma_upsample_sram2ddr


def get_gt_out(out_shape, inp, c, h_scale, w_scale, w):
    pad_widths = [(0, o0 - o1) for o0, o1 in zip(out_shape, (h_scale, w * w_scale, c))]
    assert len(inp.shape) == 2 and len(out_shape) == 3 and len(pad_widths) == 3

    inp_nhwc_np = np.expand_dims(inp[:, :c], axis=(0, 1))
    if inp.dtype in ("uint32", "uint16"):
        # Torch Tensor doesn't support numpy uint16/uint32/uint64
        inp_nhwc_np = inp_nhwc_np.astype("int64")
    inp_nchw_pt = torch.Tensor(inp_nhwc_np.transpose([0, 3, 1, 2]))

    upsample = torch.nn.Upsample(scale_factor=(h_scale, w_scale))
    ups_nchw_pt = upsample(inp_nchw_pt)
    transpose_hwc = ups_nchw_pt.permute(0, 2, 3, 1).numpy()[0, :, :, :]
    pad_hwc = np.pad(transpose_hwc, pad_widths, mode="constant", constant_values=0)
    assert pad_hwc.shape == out_shape
    return pad_hwc.astype(inp.dtype)


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_dma_upsample(dtype):
    # source
    # 2D input with WC layout
    inp_shape = rand(2, "int32", low=5, high=7, enable_corner_values=False, return_python_type=True)
    w, src_c_stride = inp_shape
    c = src_c_stride - rand(1, "int32", low=0, high=2, return_python_type=True)
    elems_inp = np.prod(inp_shape).tolist()
    inp = np.full(inp_shape, 0, dtype=dtype)
    inp[:w, :c] = np.array(range(w * c), dtype).reshape((w, c))

    # destination
    w_scale = rand(1, "int32", low=1, high=8, return_python_type=True)
    h_scale = rand(1, "int32", low=1, high=9, return_python_type=True)
    dst_c_stride = c + rand(1, "int32", low=0, high=3, return_python_type=True)
    dst_w_stride = w * w_scale + rand(1, "int32", low=0, high=3, return_python_type=True)
    out_shape = (h_scale, dst_w_stride, dst_c_stride)

    # assert mask
    assert_mask = np.full(out_shape, False, dtype=bool)
    assert_mask[:, :w, :c] = True

    gt_out = get_gt_out(out_shape, inp, c, h_scale, w_scale, w)
    bm = BuildManager()
    for direction in ("ddr_to_lsram", "lsram_to_ddr", "ddr_to_shared", "shared_to_ddr"):
        print(f"{direction}, {dtype}, inp_shape:{inp_shape}, out_shape:{out_shape}")
        prim_func = gen_dma_upsample(
            direction,
            dtype,
            elems_inp,
            h_scale,
            w_scale,
            c,
            w,
            src_c_stride,
            dst_c_stride,
            dst_w_stride,
        )
        ex = bm.build(prim_func)

        py_out = np.empty(out_shape, dtype=dtype)
        prim_func(inp, py_out)
        assert_allclose(py_out[assert_mask], gt_out[assert_mask])

        npu_out = np.empty(out_shape, dtype=dtype)
        ex(inp, npu_out)
        assert_allclose(npu_out[assert_mask], gt_out[assert_mask])


def gen_dma_upsample_nhwc(direction, dtype, n, h, h_scale, w_scale, c, w):
    inp_elems = n * h * w * c
    num_elem_upsampled = n * h * h_scale * w * w_scale * c

    @S.prim_func
    def dma_upsample_sram2ddr_nhwc(inp: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        sram = S.alloc(inp_elems, dtype, scope="lsram")
        S.dma_copy(sram, inp, inp_elems)

        for nh_idx in range(n * h):
            cur_dst = out + nh_idx * h_scale * w * w_scale * c
            cur_src = sram + nh_idx * w * c
            S.dma_upsample(cur_dst, cur_src, h_scale, w_scale, c, w)

    @S.prim_func
    def dma_upsample_ddr2sram_nhwc(inp: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        sram = S.alloc(num_elem_upsampled, dtype, scope="lsram")

        for nh_idx in range(n * h):
            cur_dst = sram + nh_idx * h_scale * w * w_scale * c
            cur_src = inp + nh_idx * w * c
            S.dma_upsample(cur_dst, cur_src, h_scale, w_scale, c, w)
        S.dma_copy(out, sram, num_elem_upsampled)

    return dma_upsample_ddr2sram_nhwc if direction[:3] == "ddr" else dma_upsample_sram2ddr_nhwc


def get_gt_out_nhwc(out_shape, inp, c, h_scale, w_scale):
    assert len(inp.shape) == len(out_shape) == 4
    inp_nchw_pt = torch.Tensor(inp.transpose([0, 3, 1, 2]))

    upsample = torch.nn.Upsample(scale_factor=(h_scale, w_scale))
    ups_nchw_pt = upsample(inp_nchw_pt)
    tranpose_nhwc = ups_nchw_pt.permute(0, 2, 3, 1).numpy()
    assert tranpose_nhwc.shape == out_shape
    return tranpose_nhwc


def test_dma_upsample_for_nhwc_input():
    dtype = "float32"

    # source
    # 4D input with NHWC layout
    inp_shape = rand(4, "int32", low=1, high=5, enable_corner_values=False, return_python_type=True)
    n, h, w, c = inp_shape
    inp = np.array(range(np.product(inp_shape)), dtype).reshape(inp_shape)

    # destination
    w_scale = rand(1, "int32", low=1, high=4, return_python_type=True)
    h_scale = rand(1, "int32", low=1, high=4, return_python_type=True)
    out_shape = (n, h * h_scale, w * w_scale, c)

    gt_out = get_gt_out_nhwc(out_shape, inp, c, h_scale, w_scale)
    bm = BuildManager()
    for direction in ("ddr_to_lsram", "lsram_to_ddr"):
        print(f"{direction}, {dtype}, inp_shape:{inp_shape}, out_shape:{out_shape}")
        prim_func = gen_dma_upsample_nhwc(direction, dtype, n, h, h_scale, w_scale, c, w)
        ex = bm.build(prim_func)

        py_out = np.empty(out_shape, dtype=dtype)
        prim_func(inp, py_out)
        assert_allclose(py_out, gt_out)

        npu_out = np.empty(out_shape, dtype=dtype)
        ex(inp, npu_out)
        assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_dma_upsample(dtype="uint32")
    test_dma_upsample_for_nhwc_input()
