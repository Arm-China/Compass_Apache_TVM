# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import os
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.tir import Aiff


@S.prim_func
def aiff_2node_conv(ctrl: S.ptr("u32", "global"), param: S.ptr("u32", "global"), act: S.ptr("u32", "global")):
    if S.get_local_id() != 0:
        return

    S.aiff(ctrl, param, act)


def get_aiff(inp, weight):
    aiff = Aiff()
    aiff.sys.status.clr_stat = 1

    aiff.mtp.twin_ctrl.mtp_0_disen = 0
    aiff.mtp.twin_ctrl.mtp_1_disen = 0
    aiff.mtp.twin_ctrl.twin_output = 1
    aiff.mtp.twin_ctrl.twin_op_mode = 2
    aiff.mtp.twin_ctrl.batch_num = 0
    aiff.mtp[0].ctrl.op_fmt = 4
    aiff.mtp[0].ctrl.ich_gsize = 3
    aiff.mtp[0].unb_ctrl.unb_mode = 1
    aiff.mtp[0].unb_ctrl.unb_pbuf_size = 3
    aiff.mtp[0].kernel_ctrl.kernel_w = 1
    aiff.mtp[0].kernel_ctrl.kernel_h = 1
    aiff.mtp[0].kernel_ctrl.stride_w = 1
    aiff.mtp[0].kernel_ctrl.stride_h = 1
    aiff.mtp[0].weight_sparse.weight_sparse = 2
    aiff.mtp[0].weight_size.weight_size = 1
    aiff.mtp[0].w_step.w_step_in = 15
    aiff.mtp[0].w_step.w_step_out = 15
    aiff.mtp[0].w_step.w_step_len = 15
    aiff.mtp[0].h_step.h_step_in = 15
    aiff.mtp[0].h_step.h_step_out = 15
    aiff.mtp[0].h_step.h_step_len = 15
    aiff.mtp[0].batch_stride = 237600
    aiff.mtp[0].iact_w_stride.iact_width_stride = 480
    aiff.mtp[0].iact_s_stride.iact_surface_stride = 7200
    aiff.mtp[0].iact_total_stride.iact_tail_stride = 29700
    aiff.mtp[0].act_c_ctrl.iact_chan = 1056
    aiff.mtp[0].act_c_ctrl.oact_chan = 64
    aiff.mtp[0].act_w_ctrl.iact_width = 15
    aiff.mtp[0].act_w_ctrl.oact_width = 15
    aiff.mtp[0].act_h_ctrl.iact_height = 15
    aiff.mtp[0].act_h_ctrl.oact_height = 15
    aiff.mtp[0].wt_compress.compression_format = 6
    aiff.mtp[0].wt_compress.wt_size = 1722
    aiff.mtp[1].ctrl.w_region_id = 1
    aiff.mtp[1].ctrl.op_fmt = 4
    aiff.mtp[1].ctrl.ich_gsize = 3
    aiff.mtp[1].unb_ctrl.unb_mode = 1
    aiff.mtp[1].unb_ctrl.unb_pbuf_size = 3
    aiff.mtp[1].kernel_ctrl.kernel_w = 1
    aiff.mtp[1].kernel_ctrl.kernel_h = 1
    aiff.mtp[1].kernel_ctrl.stride_w = 1
    aiff.mtp[1].kernel_ctrl.stride_h = 1
    aiff.mtp[1].weight_sparse.weight_sparse = 2
    aiff.mtp[1].weight_size.weight_size = 1
    aiff.mtp[1].w_step.w_step_in = 15
    aiff.mtp[1].w_step.w_step_out = 15
    aiff.mtp[1].w_step.w_step_len = 15
    aiff.mtp[1].h_step.h_step_in = 15
    aiff.mtp[1].h_step.h_step_out = 15
    aiff.mtp[1].h_step.h_step_len = 15
    aiff.mtp[1].batch_stride = 237600
    aiff.mtp[1].iact_w_stride.iact_width_stride = 480
    aiff.mtp[1].iact_s_stride.iact_surface_stride = 7200
    aiff.mtp[1].iact_total_stride.iact_tail_stride = 29700
    aiff.mtp[1].act_c_ctrl.iact_chan = 1056
    aiff.mtp[1].act_c_ctrl.oact_chan = 64
    aiff.mtp[1].act_w_ctrl.iact_width = 15
    aiff.mtp[1].act_w_ctrl.oact_width = 15
    aiff.mtp[1].act_h_ctrl.iact_height = 15
    aiff.mtp[1].act_h_ctrl.oact_height = 15
    aiff.mtp[1].wt_compress.compression_format = 6
    aiff.mtp[1].wt_compress.wt_size = 1726

    aiff.itp[0].m0_ccfg.alu_psrc = 1
    aiff.itp[0].m0_ccfg.alu_psrc_prec = 1
    aiff.itp[0].m0_ccfg.mul_trsh = 15
    aiff.itp[0].m0_ccfg.mul_psrc_dtype = 1
    aiff.itp[0].m0_ccfg.relu_byp = 1
    aiff.itp[0].m0_mul_pcfg = 189
    aiff.itp[0].m1_ccfg.byp = 1
    aiff.itp[0].e_ccfg.byp = 1
    aiff.itp[0].e_alu_clamp_pcfg0 = 2147483647
    aiff.itp[0].lut_ocfg.byp = 1
    aiff.itp[0].ocvt_mul_cfg.k = 1
    aiff.itp[0].act_ccfg0.act_height = 15
    aiff.itp[0].act_ccfg0.act_width = 15
    aiff.itp[0].act_ccfg1.act_chan = 64
    aiff.itp[0].act_ccfg1.op_mode = 1
    aiff.itp[0].act_step_ccfg.act_wstep_in = 15
    aiff.itp[0].act_step_ccfg.act_hstep_in = 15
    aiff.itp[0].act_step_ccfg.act_wstep_len = 15
    aiff.itp[0].act_step_ccfg.act_hstep_len = 15
    aiff.itp[1].m0_ccfg.alu_psrc = 1
    aiff.itp[1].m0_ccfg.alu_psrc_prec = 1
    aiff.itp[1].m0_ccfg.mul_trsh = 15
    aiff.itp[1].m0_ccfg.mul_psrc_dtype = 1
    aiff.itp[1].m0_ccfg.relu_byp = 1
    aiff.itp[1].m0_mul_pcfg = 189
    aiff.itp[1].m1_ccfg.byp = 1
    aiff.itp[1].e_ccfg.byp = 1
    aiff.itp[1].e_alu_clamp_pcfg0 = 2147483647
    aiff.itp[1].lut_ocfg.byp = 1
    aiff.itp[1].ocvt_mul_cfg.k = 1
    aiff.itp[1].act_ccfg0.act_height = 15
    aiff.itp[1].act_ccfg0.act_width = 15
    aiff.itp[1].act_ccfg1.act_chan = 64
    aiff.itp[1].act_ccfg1.op_mode = 1
    aiff.itp[1].act_step_ccfg.act_wstep_in = 15
    aiff.itp[1].act_step_ccfg.act_hstep_in = 15
    aiff.itp[1].act_step_ccfg.act_wstep_len = 15
    aiff.itp[1].act_step_ccfg.act_hstep_len = 15

    aiff.ptp.mode.mode = 7

    aiff.wrb.mode_ctrl.region_en = 3
    aiff.wrb.mode_ctrl.fmt_trans = 1
    aiff.wrb.mode_ctrl.region_out_buf = 3
    aiff.wrb.mode_ctrl.region1_size = 4
    aiff.wrb.mode_ctrl.region_bypass = 0
    aiff.wrb.region0_oact_l_stride.oact_line_stride = 38
    aiff.wrb.region0_oact_s_stride.oact_surface_stride = 570
    aiff.wrb.region0_int_ls_stride.int_surface_stride = 225
    aiff.wrb.region0_int_ls_stride.int_line_stride = 15
    aiff.wrb.region0_oact_batch_stride.oact_batch_stride = 8550
    aiff.wrb.region1_oact_l_stride.oact_line_stride = 38
    aiff.wrb.region1_oact_s_stride.oact_surface_stride = 570
    aiff.wrb.region1_int_ls_stride.int_surface_stride = 225
    aiff.wrb.region1_int_ls_stride.int_line_stride = 15
    aiff.wrb.region1_oact_batch_stride.oact_batch_stride = 8550

    aiff.unb.mtp1_base_addr.base_addr = 4096
    aiff.unb.itp_base_addr.base_addr = 8192

    aiff.mtp[0].weight_addr.weight_addr = weight[608:]
    aiff.mtp[1].weight_addr.weight_addr = weight[55712:]
    aiff.itp[0].m0_alu_mcfg = weight
    aiff.itp[1].m0_alu_mcfg = weight[128:]
    aiff.mtp[0].iact_addr = inp
    aiff.mtp[1].iact_addr = inp

    aiff.add_new_register_config(copy_idx=0)

    aiff.mtp[0].act_c_ctrl.iact_chan = 1056
    aiff.mtp[0].act_c_ctrl.oact_chan = 96
    aiff.mtp[0].wt_compress.compression_format = 6
    aiff.mtp[0].wt_compress.wt_size = 2598
    aiff.mtp[1].act_c_ctrl.iact_chan = 1056
    aiff.mtp[1].act_c_ctrl.oact_chan = 80
    aiff.mtp[1].wt_compress.compression_format = 6
    aiff.mtp[1].wt_compress.wt_size = 2550

    aiff.itp[0].act_ccfg1.act_chan = 96
    aiff.itp[0].act_ccfg1.op_mode = 1
    aiff.itp[1].act_ccfg1.act_chan = 80
    aiff.itp[1].act_ccfg1.op_mode = 1

    aiff.unb.mtp1_base_addr.base_addr = 6144
    aiff.unb.itp_base_addr.base_addr = 12288

    aiff.mtp[0].weight_addr.weight_addr = weight[110944:]
    aiff.mtp[1].weight_addr.weight_addr = weight[194080:]
    aiff.itp[0].m0_alu_mcfg = weight[256:]
    aiff.itp[1].m0_alu_mcfg = weight[448:]
    aiff.mtp[0].iact_addr = inp
    aiff.mtp[1].iact_addr = inp

    return aiff


def get_desc(aiff, out):
    aiff.reg_cfgs[0].wrb.region0_oact_addr = out
    aiff.reg_cfgs[0].wrb.region1_oact_addr = out[:, :, :, 64:]
    aiff.reg_cfgs[1].wrb.region0_oact_addr = out[:, :, :, 128:]
    aiff.reg_cfgs[1].wrb.region1_oact_addr = out[:, :, :, 224:]

    return aiff.gen_descriptor()


@pytest.mark.X2_1204
def test_aiff_2node_conv():
    dtype = "int8"
    out_shape = (1, 15, 15, 304)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(cur_dir, "../../../data/aiff_2node_conv"))
    inp = np.fromfile(f"{data_dir}/input0.bin", dtype="int8").reshape([1, 33, 15, 15, 32])
    weight = np.fromfile(f"{data_dir}/weight.bin", dtype="int8")
    gt_out = np.fromfile(f"{data_dir}/gt.bin", dtype="int8").reshape(out_shape)
    aiff = get_aiff(inp, weight)

    bm = aipu.tir.BuildManager()
    ex = bm.build(aiff_2node_conv)

    py_out = np.empty(out_shape, dtype=dtype)
    desc = get_desc(aiff, py_out)
    aiff_2node_conv(desc.ctrl, desc.param, desc.act)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(out_shape, dtype=dtype)
    desc = get_desc(aiff, aipu_out)
    ex(desc.ctrl, desc.param, desc.act)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_aiff_2node_conv()
