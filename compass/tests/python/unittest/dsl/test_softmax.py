# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import torch
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


MAX_VECS_ON_LSRAM = 992


def gen_softmax_fast_exp(dtype):
    type_min = np.iinfo(dtype).min
    vdtype = hw_native_vdtype(dtype)
    lanes = vdtype.lanes
    out_dtype = vdtype.with_uint().element_of

    @S.prim_func
    def line_max_value(input_addr: S.ptr(vdtype, "lsram"), width: S.i32) -> S.i16x16:
        min_value = S.cast(type_min, vdtype)
        loop = width / lanes
        tmp = width - loop * lanes
        b_linemax = min_value

        for i in range(loop):
            b_linemax = S.max(b_linemax, input_addr[i])
        if tmp > 0:
            b_linemax = S.max(b_linemax, input_addr[loop], mask=S.tail_mask(tmp, lanes), r=b_linemax)
        # need get a line max to vector whose element is maxvalue
        b_linemax = S.vmaxh(b_linemax, b_linemax)
        b_linemax = S.vmaxh(b_linemax, b_linemax)
        b_linemax = S.vmaxh(b_linemax, b_linemax)
        b_linemax = S.vmaxh(b_linemax, b_linemax)
        b_linemax = S.vmaxh(b_linemax, b_linemax)
        return S.vxtl(b_linemax)

    @S.prim_func
    def log2_norm(
        uw_lut_sum: S.ptr("u32x8"),
        pow2_out_q: S.i32,
        adjust_q: S.i16,
    ) -> S.i32x8:
        uw_n0 = S.clz(uw_lut_sum[0]) - 1
        uw_lut_sum[0] = uw_lut_sum[0] << uw_n0

        w_compt_and_p = S.vand(uw_lut_sum[0] > S.u32(1518500249), adjust_q != 0)
        w_tmp_lsram = S.vsl(S.i(uw_n0) - (30 - pow2_out_q), adjust_q, saturate=True)
        return S.vsel(S.i32x8(1), 0, w_compt_and_p) - w_tmp_lsram

    @S.prim_func
    def compute(
        b_inout_lsram: S.ptr(dtype, "lsram"),
        output_addr: S.ptr(out_dtype, "lsram"),
        local_w32_addr: S.ptr("i32", "lsram"),
        each_t_num: S.i32,
        width: S.i32,
        scale: S.i16,
        shift: S.i32,
        pow2_out_q: S.i32,
        adjust_q: S.i32,
    ):
        loop16 = width / 16
        tail_tmp16 = width - loop16 * 16
        tmp_less_8 = S.max(0, tail_tmp16 - 8)

        loop8 = width / 8
        tail_tmp8 = width - loop8 * 8

        out_addr_index = 0
        for i in range(each_t_num):
            input_addr_index = i * width
            # step 1: max_v, _ = x.max(axis, keepdim=True), max_v is 16-bit
            h_linemax = line_max_value(b_inout_lsram + input_addr_index, width)
            log2ex_addr = 0
            uw_lut_sum = S.u32x8(0)
            # one loop can get 16 log2ex, and accumulate 16 times
            for j in range(loop16):
                b_input = S.vload(b_inout_lsram + input_addr_index + 16 * j, "16T")
                # log2e_x = x-max_v
                h_index = S.vxtl(b_input) - h_linemax
                # log2e_x = (log2e_x*input_scale_value)>>(input_shift_value)
                w_output = S.vmull(h_index, scale) >> shift
                # save log2e_x(32bit) for future usage after log2_norm
                S.vstore(w_output, local_w32_addr + log2ex_addr)
                log2ex_addr += 8
                # y = (1<<(Pow2OutQ+log2e_x))
                # y_sum = y.sum(axis, keepdim=True, dtype=torch.long)
                uw_lut_sum += S.u(1 << (w_output + pow2_out_q))
                # same as the above, calc high 16*8
                w_output = S.vmulh(h_index, scale) >> shift
                S.vstore(w_output, local_w32_addr + log2ex_addr)
                log2ex_addr += 8
                uw_lut_sum += S.u(1 << (w_output + pow2_out_q))
            # process tail, calc process is same
            if tail_tmp16 > 0:
                tail4 = S.min(8, tail_tmp16)
                w_tail4_p = S.tail_mask(tail4, 8)
                b_input = S.vload(b_inout_lsram + input_addr_index + 16 * loop16, S.tail_mask(tail_tmp16, 32))
                h_index = S.vxtl(b_input) - h_linemax
                w_output = S.vmull(h_index, scale) >> shift
                S.vstore(w_output, local_w32_addr + log2ex_addr, w_tail4_p)
                log2ex_addr += tail4
                uw_output = 1 << (w_output + pow2_out_q)
                uw_lut_sum = S.vadd(uw_lut_sum, uw_output, out_sign="u", mask=w_tail4_p, r=uw_lut_sum)
                if tmp_less_8 > 0:
                    w_tail42_p = S.tail_mask(tmp_less_8, 8)
                    w_output = S.vmulh(h_index, scale) >> shift
                    S.vstore(w_output, local_w32_addr + log2ex_addr, w_tail42_p)
                    uw_output = 1 << (w_output + pow2_out_q)
                    uw_lut_sum = S.vadd(uw_lut_sum, uw_output, out_sign="u", mask=w_tail42_p, r=uw_lut_sum)
            # y_sum = y.sum(axis, keepdim=True, dtype=torch.long)
            uw_lut_sum = S.vaddh(uw_lut_sum, uw_lut_sum)
            uw_lut_sum = S.vaddh(uw_lut_sum, uw_lut_sum)
            uw_lut_sum = S.vaddh(uw_lut_sum, uw_lut_sum)
            # judge if 1<y_sum<2**31
            uw_lut_sum = S.min(S.max(uw_lut_sum, 1), (1 << 31) - 1)
            w_lut_out = log2_norm(uw_lut_sum.addr, pow2_out_q, adjust_q)
            log2ex_addr = 0
            # yy_div_sum = (((log2e_x+out.qbits)<<adjust_q) - log2_sum)>>adjust_q
            # out= (1<<yy_div_sum)
            for j in range(loop8):
                w_xmax = S.vload(local_w32_addr + log2ex_addr)
                uw_out = S.u32x8(1) << ((((w_xmax + 8) << adjust_q) - w_lut_out) >> adjust_q)
                ub_result = S.vcompt(S.vnsr(uw_out, 0, saturate=True), "8TFFF")
                S.vstore(ub_result, output_addr + out_addr_index, "8T")
                log2ex_addr = log2ex_addr + 8
                out_addr_index = out_addr_index + 8
            if tail_tmp8 > 0:
                w_xmax = S.vload(local_w32_addr + log2ex_addr, S.tail_mask(tail_tmp8, 8))
                uw_out = S.u32x8(1) << ((((w_xmax + 8) << adjust_q) - w_lut_out) >> adjust_q)
                ub_result = S.vcompt(S.vnsr(uw_out, 0, saturate=True), "8TFFF")
                S.vstore(ub_result, output_addr + out_addr_index, S.tail_mask(tail_tmp8, 32))
                out_addr_index = out_addr_index + tail_tmp8

    @S.prim_func(is_entry=True)
    def softmax_fast_exp(
        input_addr: S.ptr(dtype, "global"),
        output_addr: S.ptr(out_dtype, "global"),
        height: S.i32,
        width: S.i32,
        scale: S.i16,
        shift: S.i32,
        pow2_out_q: S.i32,
        adjust_q: S.i32,
    ):
        tec_num = S.get_local_size()
        tid = S.get_local_id()
        each_t_num = (MAX_VECS_ON_LSRAM * 32 - width * 4) / (width * 2)
        each_pe_num = each_t_num * width
        for_first_loop = height / (each_t_num * tec_num)
        tot_sram_buffer = S.alloc_buffer((MAX_VECS_ON_LSRAM * 32,), dtype=dtype, scope="lsram")
        in_sram = tot_sram_buffer.addr_of(0)
        out_sram = tot_sram_buffer.addr_of(each_pe_num).as_ptr(out_dtype)
        tmp_sram = tot_sram_buffer.addr_of(each_pe_num * 2).as_ptr("i32")
        for i in range(for_first_loop):
            base = i * each_pe_num * tec_num + tid * each_pe_num
            S.dma_copy(in_sram, input_addr + base, each_pe_num)
            compute(in_sram, out_sram, tmp_sram, each_t_num, width, scale, shift, pow2_out_q, adjust_q)
            S.dma_copy(output_addr + base, out_sram, each_pe_num)
        line_tail = height - for_first_loop * each_t_num * tec_num
        for_second_loop = line_tail / tec_num
        each_pe_num_sec = for_second_loop * width

        if for_second_loop > 0:
            addr_in_out_d0 = for_first_loop * each_pe_num * tec_num + tid * each_pe_num_sec
            S.dma_copy(in_sram, input_addr + addr_in_out_d0, each_pe_num_sec)
            compute(in_sram, out_sram, tmp_sram, for_second_loop, width, scale, shift, pow2_out_q, adjust_q)
            S.dma_copy(output_addr + addr_in_out_d0, out_sram, each_pe_num_sec)
        last_line = line_tail - for_second_loop * tec_num
        if last_line > 0:
            addr_in_out_d0 = for_first_loop * each_pe_num * tec_num + each_pe_num_sec * tec_num + tid * width
            if tid < last_line:
                S.dma_copy(in_sram, input_addr + addr_in_out_d0, width)
                compute(in_sram, out_sram, tmp_sram, 1, width, scale, shift, pow2_out_q, adjust_q)
                S.dma_copy(output_addr + addr_in_out_d0, out_sram, width)

    return softmax_fast_exp


def simple_log2(tx, loginQ, logoutQ):
    vshape = tx.shape
    x = tx.reshape(-1)
    var_out = torch.zeros_like(x)
    # search counting the leading zero of the value with asm instruction(clz)
    for _ in range(0, 30):
        cond = x < 0x40000000
        var_out = torch.where(cond, var_out + 1, var_out)
        x = torch.where(cond, x << 1, x)
    if logoutQ == 0:
        y = torch.zeros_like(x)
    else:
        # this Magic number is float sqrt(2)/2 *2^31
        y = (x > 1518500250).int()
    y = y - ((var_out - (30 - loginQ)) << logoutQ)

    return y.reshape(vshape)


def softmax_gt(input_data, input_scale_value, input_shift_value, out_qbits, adjust_q, pow2_out_qvalue, axis):
    maxvalue = (1 << out_qbits) - 1
    x = input_data.int()
    max_v, _ = x.max(axis, keepdim=True)
    log2e_x = (x - max_v).long()
    log2e_x = (log2e_x * input_scale_value) >> (input_shift_value)
    # *((pow2_out_qvalue+log2e_x)>=0) to fix torch version diff issue,lib neednot it
    y = (1 << (pow2_out_qvalue + log2e_x)) * ((pow2_out_qvalue + log2e_x) >= 0)
    y_sum = y.sum(axis, keepdim=True, dtype=torch.long)
    yy_sum = torch.clamp(y_sum, 1, (1 << 31) - 1)
    log2_sum = simple_log2(yy_sum, pow2_out_qvalue, adjust_q)
    yy_div_sum = (((log2e_x + out_qbits) << adjust_q) - log2_sum) >> adjust_q
    # *(yy_div_sum>=0) to fix torch version diff issue,lib neednot it
    out = (1 << yy_div_sum) * (yy_div_sum >= 0)
    out = torch.clamp(out, 0, maxvalue)
    return out.numpy().astype(np.uint8)


@pytest.mark.parametrize("dtype", ("int8",))
def test_softmax(dtype):
    x_2d_shape = rand((2), dtype="int32", low=1, high=1024, enable_corner_values=False)
    x = rand(list(x_2d_shape), dtype)
    x_scale_value = rand(1, dtype="int32", low=128, high=32767, enable_corner_values=False)
    x_shift_value = rand(1, dtype="int32", low=0, high=15, enable_corner_values=False)
    adjust_q = rand(1, dtype="int32", low=0, high=2, enable_corner_values=False)
    pow2_out_qvalue = rand(1, dtype="int32", low=10, high=30, enable_corner_values=False)
    out_qbits = 8
    axis = 1
    height, width = x_2d_shape[0], x_2d_shape[1]
    bm = BuildManager()
    py_func = gen_softmax_fast_exp(dtype)
    ex = bm.build(py_func)

    gt_out = softmax_gt(
        torch.tensor(x),
        x_scale_value,
        x_shift_value,
        out_qbits,
        adjust_q,
        pow2_out_qvalue,
        axis,
    )
    py_out = np.empty((height, width), dtype="uint8")
    py_func(x, py_out, height, width, x_scale_value, x_shift_value, pow2_out_qvalue, adjust_q)
    assert_allclose(gt_out, py_out)

    npu_out = np.empty((height, width), dtype="uint8")
    ex(x, npu_out, height, width, x_scale_value, x_shift_value, pow2_out_qvalue, adjust_q)
    assert_allclose(gt_out, npu_out)


if __name__ == "__main__":
    test_softmax("int8")
