# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=missing-class-docstring
"""The X2 specific part of AIFF register."""
from ..register import FieldInfo, Register


#
# System Registers
#
class AiffCtrl(Register):
    field_infos = {"act_wt_route": FieldInfo(8, 8)}

    def __init__(self):
        self.act_wt_route = 0
        super().__init__(0x000)


class AiffStatus(Register):
    field_infos = {"clr_stat": FieldInfo(31, 31)}

    def __init__(self):
        self.clr_stat = 0
        super().__init__(0x001)


#
# Unified Buffer (UnB) Registers
#
class UnbMtp0BaseAddr(Register):
    field_infos = {"base_addr": FieldInfo(4, 19)}

    def __init__(self):
        self.base_addr = 0
        super().__init__(0x020)


class UnbMtp1BaseAddr(Register):
    field_infos = {"base_addr": FieldInfo(4, 19)}

    def __init__(self):
        self.base_addr = 0
        super().__init__(0x021)


class UnbItpBaseAddr(Register):
    field_infos = {"base_addr": FieldInfo(4, 19)}

    def __init__(self):
        self.base_addr = 0
        super().__init__(0x022)


class UnbPtpBaseAddr(Register):
    field_infos = {"base_addr": FieldInfo(4, 19)}

    def __init__(self):
        self.base_addr = 0
        super().__init__(0x023)


#
# Write Buffer (WrB) Registers
#
class WrbRegion0OactAddr(Register):
    field_infos = {"oact_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "act"

    def __init__(self):
        self.oact_addr = 0
        super().__init__(0x030, 40)


class WrbRegion0OactCtagAddr(Register):
    field_infos = {"oact_ctag_addr": FieldInfo(0, 31)}
    desc_type = "act"

    def __init__(self):
        self.oact_ctag_addr = 0
        super().__init__(0x031, 41)


class WrbRegion1OactAddr(Register):
    field_infos = {"oact_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "act"

    def __init__(self):
        self.oact_addr = 0
        super().__init__(0x032, 42)


class WrbRegion1OactCtagAddr(Register):
    field_infos = {"oact_ctag_addr": FieldInfo(0, 31)}
    desc_type = "act"

    def __init__(self):
        self.oact_ctag_addr = 0
        super().__init__(0x033, 43)


class WrbModeCtrl(Register):
    field_infos = {
        "region_en": FieldInfo(0, 1),
        "region_ppen": FieldInfo(5, 5),
        "fmt_trans": FieldInfo(8, 8),
        "region_out_buf": FieldInfo(12, 13),
        "region_int_buf": FieldInfo(16, 17),
        "region1_size": FieldInfo(20, 23),
        "region0_alloc": FieldInfo(24, 24),
        "region1_alloc": FieldInfo(25, 25),
        "region_bypass": FieldInfo(28, 29),
    }

    def __init__(self):
        self.region_en = 0
        self.region_ppen = 0
        self.fmt_trans = 0
        self.region_out_buf = 0
        self.region_int_buf = 0
        self.region1_size = 0
        self.region0_alloc = 0
        self.region1_alloc = 0
        self.region_bypass = 3
        super().__init__(0x040)


class WrbRegion0OactCtrl(Register):
    field_infos = {"oact_enhanced_sparse": FieldInfo(1, 1)}

    def __init__(self):
        self.oact_enhanced_sparse = 0
        super().__init__(0x041)


class WrbRegion0OactLStride(Register):
    field_infos = {"oact_line_stride": FieldInfo(3, 24)}

    def __init__(self):
        self.oact_line_stride = 0
        super().__init__(0x042)


class WrbRegion0OactSStride(Register):
    field_infos = {"oact_surface_stride": FieldInfo(3, 30)}

    def __init__(self):
        self.oact_surface_stride = 0
        super().__init__(0x043)


class WrbRegion0IntLsStride(Register):
    field_infos = {
        "int_surface_stride": FieldInfo(0, 11),
        "int_line_stride": FieldInfo(16, 26),
    }

    def __init__(self):
        self.int_surface_stride = 0
        self.int_line_stride = 0
        super().__init__(0x044)


class WrbRegion0OctagSStride(Register):
    field_infos = {"octag_surface_stride": FieldInfo(0, 30)}

    def __init__(self):
        self.octag_surface_stride = 0
        super().__init__(0x045)


class WrbRegion0OactBatchStride(Register):
    field_infos = {"oact_batch_stride": FieldInfo(3, 31)}

    def __init__(self):
        self.oact_batch_stride = 0
        super().__init__(0x046)


class WrbRegion0OctagBatchStride(Register):
    field_infos = {"octag_batch_stride": FieldInfo(0, 31)}

    def __init__(self):
        self.octag_batch_stride = 0
        super().__init__(0x047)


class WrbRegion0OactWhOffset(Register):
    field_infos = {"oact_wh_offset": FieldInfo(5, 30)}

    def __init__(self):
        self.oact_wh_offset = 0
        super().__init__(0x048)


# This fake register is used to meet the alignment requirement of control descriptor, the base
# address in control descriptor need to be multiple of 8.
class WrbRegion1Placeholder(Register):
    field_infos = {"useless": FieldInfo(0, 31)}

    def __init__(self):
        self.useless = 0
        super().__init__(0x050)


class WrbRegion1OactCtrl(Register):
    field_infos = {"oact_enhanced_sparse": FieldInfo(1, 1)}

    def __init__(self):
        self.oact_enhanced_sparse = 0
        super().__init__(0x051)


class WrbRegion1OactLStride(Register):
    field_infos = {"oact_line_stride": FieldInfo(3, 24)}

    def __init__(self):
        self.oact_line_stride = 0
        super().__init__(0x052)


class WrbRegion1OactSStride(Register):
    field_infos = {"oact_surface_stride": FieldInfo(3, 30)}

    def __init__(self):
        self.oact_surface_stride = 0
        super().__init__(0x053)


class WrbRegion1IntLsStride(Register):
    field_infos = {
        "int_surface_stride": FieldInfo(0, 11),
        "int_line_stride": FieldInfo(16, 26),
    }

    def __init__(self):
        self.int_surface_stride = 0
        self.int_line_stride = 0
        super().__init__(0x054)


class WrbRegion1OctagSStride(Register):
    field_infos = {"octag_surface_stride": FieldInfo(0, 30)}

    def __init__(self):
        self.octag_surface_stride = 0
        super().__init__(0x055)


class WrbRegion1OactBatchStride(Register):
    field_infos = {"oact_batch_stride": FieldInfo(3, 31)}

    def __init__(self):
        self.oact_batch_stride = 0
        super().__init__(0x056)


class WrbRegion1OctagBatchStride(Register):
    field_infos = {"octag_batch_stride": FieldInfo(0, 31)}

    def __init__(self):
        self.octag_batch_stride = 0
        super().__init__(0x057)


class WrbRegion1OactWhOffset(Register):
    field_infos = {"oact_wh_offset": FieldInfo(5, 30)}

    def __init__(self):
        self.oact_wh_offset = 0
        super().__init__(0x058)


#
# Matrix-to-Tensor Processing (MTP) Registers
#
class MtpTwinCtrl(Register):
    field_infos = {
        "mtp_0_disen": FieldInfo(0, 0),
        "mtp_1_disen": FieldInfo(1, 1),
        "twin_output": FieldInfo(2, 3),
        "twin_op_mode": FieldInfo(4, 5),
        "batch_num": FieldInfo(8, 9),
    }

    def __init__(self):
        self.mtp_0_disen = 0
        self.mtp_1_disen = 0
        self.twin_output = 0
        self.twin_op_mode = 0
        self.batch_num = 0
        super().__init__(0x170)


# Each MTP has a copy of the following registers.
class MtpIactAddr(Register):
    field_infos = {"iact_addr": FieldInfo(0, 31)}
    desc_type = "act"

    def __init__(self, idx):
        self.iact_addr = 0
        super().__init__(0x100 + idx * 0x080, 0 + idx * 8)


class MtpBatchMatmulIactAddr(Register):
    field_infos = {"iact_addr": FieldInfo(0, 31)}
    desc_type = "act"

    def __init__(self, idx):
        self.iact_addr = 0
        super().__init__(0x101 + idx * 0x080, 1 + idx * 8)


class MtpIactCtagAddr(Register):
    field_infos = {"ctag_base_addr": FieldInfo(0, 31)}
    desc_type = "act"

    def __init__(self, idx):
        self.ctag_base_addr = 0
        super().__init__(0x102 + idx * 0x080, 2 + idx * 8)


class MtpUnbAddr(Register):
    field_infos = {"ext_base_addr": FieldInfo(0, 31)}
    desc_type = "act"

    def __init__(self, idx):
        self.ext_base_addr = 0
        super().__init__(0x103 + idx * 0x080, 3 + idx * 8)


class MtpWeightAddr(Register):
    field_infos = {"weight_addr": FieldInfo(3, 31)}
    desc_type = "param"

    def __init__(self, idx):
        self.weight_addr = 0
        super().__init__(0x110 + idx * 0x080, 0 + idx * 8)


class MtpPadAddr(Register):
    field_infos = {"pad_addr": FieldInfo(3, 31)}
    desc_type = "param"

    def __init__(self, idx):
        self.pad_addr = 0
        super().__init__(0x111 + idx * 0x080, 1 + idx * 8)


class MtpIactFpParamAddr0(Register):
    field_infos = {"fp_addr": FieldInfo(0, 31)}
    desc_type = "param"

    def __init__(self, idx):
        self.fp_addr = 0
        super().__init__(0x112 + idx * 0x080, 2 + idx * 8)


class MtpIactFpParamAddr1(Register):
    field_infos = {"fp_addr": FieldInfo(0, 31)}
    desc_type = "param"

    def __init__(self, idx):
        self.fp_addr = 0
        super().__init__(0x113 + idx * 0x080, 3 + idx * 8)


class MtpCtrl(Register):
    field_infos = {
        "op_mode": FieldInfo(0, 2),
        "pad_en": FieldInfo(4, 4),
        "pad_mode": FieldInfo(5, 5),
        "w_region_id": FieldInfo(6, 6),
        "down_sample": FieldInfo(7, 7),
        "op_fmt": FieldInfo(8, 11),
        "ich_gsize": FieldInfo(12, 14),
        "chw_mode": FieldInfo(15, 15),
        "matmul_mode": FieldInfo(16, 18),
        "weight_reuse": FieldInfo(19, 19),
        "winograd_ctrl": FieldInfo(24, 24),
        "deconv_en": FieldInfo(25, 25),
        "input_convert": FieldInfo(26, 27),
        "input_fp_param_type": FieldInfo(28, 28),
    }

    def __init__(self, idx):
        self.op_mode = 0
        self.pad_en = 0
        self.pad_mode = 0
        self.w_region_id = 0
        self.down_sample = 0
        self.op_fmt = 0
        self.ich_gsize = 0
        self.chw_mode = 0
        self.matmul_mode = 0
        self.weight_reuse = 0
        self.winograd_ctrl = 0
        self.deconv_en = 0
        self.input_convert = 0
        self.input_fp_param_type = 0
        super().__init__(0x120 + idx * 0x080)


class MtpBeCtrl(Register):
    field_infos = {"ocvt_trsh": FieldInfo(0, 5)}

    def __init__(self, idx):
        self.ocvt_trsh = 0
        super().__init__(0x121 + idx * 0x080)


class MtpUnbCtrl(Register):
    field_infos = {
        "unb_mode": FieldInfo(0, 0),
        "unb_pbuf_size": FieldInfo(2, 3),
    }

    def __init__(self, idx):
        self.unb_mode = 0
        self.unb_pbuf_size = 3
        super().__init__(0x122 + idx * 0x080)


class MtpKernelCtrl(Register):
    field_infos = {
        "kernel_w": FieldInfo(0, 3),
        "kernel_h": FieldInfo(4, 7),
        "stride_w": FieldInfo(8, 10),
        "stride_h": FieldInfo(12, 14),
        "dilation_w": FieldInfo(16, 18),
        "dilation_h": FieldInfo(20, 22),
    }

    def __init__(self, idx):
        self.kernel_w = 0
        self.kernel_h = 0
        self.stride_w = 0
        self.stride_h = 0
        self.dilation_w = 0
        self.dilation_h = 0
        super().__init__(0x123 + idx * 0x080)


class MtpWeightSparse(Register):
    field_infos = {
        "weight_sparse": FieldInfo(0, 1),
        "block0_len": FieldInfo(4, 19),
    }

    def __init__(self, idx):
        self.weight_sparse = 0
        self.block0_len = 0
        super().__init__(0x124 + idx * 0x080)


class MtpWeightSize(Register):
    field_infos = {"weight_size": FieldInfo(0, 7)}

    def __init__(self, idx):
        self.weight_size = 0
        super().__init__(0x125 + idx * 0x080)


class MtpWStep(Register):
    field_infos = {
        "w_step_in": FieldInfo(0, 7),
        "w_step_out": FieldInfo(8, 15),
        "w_step_len": FieldInfo(16, 23),
        "w_step_ovlap": FieldInfo(24, 27),
    }

    def __init__(self, idx):
        self.w_step_in = 0
        self.w_step_out = 0
        self.w_step_len = 0
        self.w_step_ovlap = 0
        super().__init__(0x126 + idx * 0x080)


class MtpHStep(Register):
    field_infos = {
        "h_step_in": FieldInfo(0, 7),
        "h_step_out": FieldInfo(8, 15),
        "h_step_len": FieldInfo(16, 23),
        "h_step_ovlap": FieldInfo(24, 27),
    }

    def __init__(self, idx):
        self.h_step_in = 0
        self.h_step_out = 0
        self.h_step_len = 0
        self.h_step_ovlap = 0
        super().__init__(0x127 + idx * 0x080)


class MtpBatchStride(Register):
    field_infos = {"batch_stride": FieldInfo(0, 31)}

    def __init__(self, idx):
        self.batch_stride = 0
        super().__init__(0x128 + idx * 0x080)


class MtpDeconvPad(Register):
    field_infos = {
        "deconv_pad_w": FieldInfo(0, 0),
        "deconv_pad_h": FieldInfo(4, 4),
        "deconv_zero_skip_en": FieldInfo(8, 8),
    }

    def __init__(self, idx):
        self.deconv_pad_w = 0
        self.deconv_pad_h = 0
        self.deconv_zero_skip_en = 0
        super().__init__(0x129 + idx * 0x080)


class MtpIactCtagSurfStride(Register):
    field_infos = {"ctag_surface_stride": FieldInfo(0, 25)}

    def __init__(self, idx):
        self.ctag_surface_stride = 0
        super().__init__(0x12A + idx * 0x080)


class MtpIactCtagBatchStride(Register):
    field_infos = {"ctag_batch_stride": FieldInfo(0, 31)}

    def __init__(self, idx):
        self.ctag_batch_stride = 0
        super().__init__(0x12B + idx * 0x080)


class MtpIactCtrl(Register):
    field_infos = {
        "iact_src": FieldInfo(0, 0),
        "iact_wake_id": FieldInfo(1, 1),
        "act_zero_skip_en": FieldInfo(6, 6),
        "iact_sparse": FieldInfo(4, 4),
        "iact_enhanced_sparse": FieldInfo(5, 5),
    }

    def __init__(self, idx):
        self.iact_src = 0
        self.iact_wake_id = 0
        self.act_zero_skip_en = 0
        self.iact_sparse = 0
        self.iact_enhanced_sparse = 0
        super().__init__(0x12C + idx * 0x080)


class MtpPadSize(Register):
    field_infos = {
        "pad_right": FieldInfo(0, 5),
        "pad_left": FieldInfo(8, 13),
        "pad_bottom": FieldInfo(16, 21),
        "pad_top": FieldInfo(24, 29),
    }

    def __init__(self, idx):
        self.pad_right = 0
        self.pad_left = 0
        self.pad_bottom = 0
        self.pad_top = 0
        super().__init__(0x12D + idx * 0x080)


class MtpPadValue(Register):
    field_infos = {"pad_value": FieldInfo(16, 31)}

    def __init__(self, idx):
        self.pad_value = 0
        super().__init__(0x12E + idx * 0x080)


class MtpFpParam(Register):
    field_infos = {
        "act0_shift": FieldInfo(0, 7),
        "act0_scale": FieldInfo(8, 15),
        "act1_shift": FieldInfo(16, 23),
        "act1_scale": FieldInfo(24, 31),
    }

    def __init__(self, idx):
        self.act0_shift = 0
        self.act0_scale = 0
        self.act1_shift = 0
        self.act1_scale = 0
        super().__init__(0x12F + idx * 0x080)


class MtpIactWStride(Register):
    field_infos = {"iact_width_stride": FieldInfo(0, 30)}

    def __init__(self, idx):
        self.iact_width_stride = 0
        super().__init__(0x130 + idx * 0x080)


class MtpIactSStride(Register):
    field_infos = {"iact_surface_stride": FieldInfo(0, 30)}

    def __init__(self, idx):
        self.iact_surface_stride = 0
        super().__init__(0x131 + idx * 0x080)


class MtpIactTotalStride(Register):
    field_infos = {"iact_tail_stride": FieldInfo(3, 31)}

    def __init__(self, idx):
        self.iact_tail_stride = 0
        super().__init__(0x132 + idx * 0x080)


class MtpActCCtrl(Register):
    field_infos = {
        "iact_chan": FieldInfo(0, 11),
        "oact_chan": FieldInfo(16, 28),
        "iact_chan_h": FieldInfo(30, 30),
    }

    def __init__(self, idx):
        self.iact_chan = 0
        self.oact_chan = 0
        self.iact_chan_h = 0
        super().__init__(0x133 + idx * 0x080)


class MtpActWCtrl(Register):
    field_infos = {
        "iact_width": FieldInfo(0, 12),
        "oact_width": FieldInfo(16, 28),
    }

    def __init__(self, idx):
        self.iact_width = 0
        self.oact_width = 0
        super().__init__(0x134 + idx * 0x080)


class MtpActHCtrl(Register):
    field_infos = {
        "iact_height": FieldInfo(0, 11),
        "oact_height": FieldInfo(16, 27),
    }

    def __init__(self, idx):
        self.iact_height = 0
        self.oact_height = 0
        super().__init__(0x135 + idx * 0x080)


class MtpWtCompress(Register):
    field_infos = {
        "compression_format": FieldInfo(0, 3),
        "wt_size": FieldInfo(4, 31),
    }

    def __init__(self, idx):
        self.compression_format = 0
        self.wt_size = 0
        super().__init__(0x136 + idx * 0x080)


class MtpBatchMatmul0(Register):
    field_infos = {
        "bm_wt_ci_tail_size": FieldInfo(0, 15),
        "bm_wt_co_tail_size": FieldInfo(16, 31),
    }

    def __init__(self, idx):
        self.bm_wt_ci_tail_size = 0
        self.bm_wt_co_tail_size = 0
        super().__init__(0x137 + idx * 0x080)


class MtpBatchMatmul1(Register):
    field_infos = {"bm_wt_co_ci_tail_size": FieldInfo(0, 15)}

    def __init__(self, idx):
        self.bm_wt_co_ci_tail_size = 0
        super().__init__(0x138 + idx * 0x080)


class MtpBatchMatmul2(Register):
    field_infos = {"bm_wt_tail_stride": FieldInfo(0, 17)}

    def __init__(self, idx):
        self.bm_wt_tail_stride = 0
        super().__init__(0x139 + idx * 0x080)


class MtpConv3dCtrl(Register):
    field_infos = {
        "conv3d_en": FieldInfo(0, 0),
        "conv3d_depth": FieldInfo(4, 13),
    }

    def __init__(self, idx):
        self.conv3d_en = 0
        self.conv3d_depth = 0
        super().__init__(0x13A + idx * 0x080)


class MtpConv3dWtStride(Register):
    field_infos = {"conv3d_wt_stride": FieldInfo(0, 30)}

    def __init__(self, idx):
        self.conv3d_wt_stride = 0
        super().__init__(0x13B + idx * 0x080)


#
# Intra-Tensor and Inter-Tensor Processing (ITP) Registers
#
class ItpAccIntDstMcfg0(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "act"

    def __init__(self):
        self.base_addr = 0
        super().__init__(0x200, 16)


class ItpAccIntDstMcfg1(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "act"

    def __init__(self):
        self.base_addr = 0
        super().__init__(0x201, 17)


class ItpIntpMcfg(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "act"

    def __init__(self):
        self.base_addr = 0
        super().__init__(0x202, 18)


class ItpLutMcfg(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "param"

    def __init__(self):
        self.base_addr = 0
        super().__init__(0x210, 62)


class ItpModeCcfg(Register):
    field_infos = {
        "lut_glb_mode": FieldInfo(0, 0),
        "loop2_to_concat": FieldInfo(1, 1),
        "loop_en": FieldInfo(2, 2),
        "batch_num": FieldInfo(8, 9),
        "lut0_ft_en": FieldInfo(16, 16),
        "lut1_ft_en": FieldInfo(17, 17),
        "lut2_ft_en": FieldInfo(18, 18),
        "intp_dtype": FieldInfo(23, 23),
        "lin_int_en": FieldInfo(24, 24),
        "lin_int_mode": FieldInfo(25, 25),
        "rdc_en": FieldInfo(26, 26),
    }

    def __init__(self):
        self.lut_glb_mode = 0
        self.loop2_to_concat = 0
        self.loop_en = 0
        self.batch_num = 0
        self.lut0_ft_en = 0
        self.lut1_ft_en = 0
        self.lut2_ft_en = 0
        self.intp_dtype = 0
        self.lin_int_en = 0
        self.lin_int_mode = 0
        self.rdc_en = 0
        super().__init__(0x220)


class ItpUnbAddrMcfg(Register):
    field_infos = {
        "temp_cube_size": FieldInfo(16, 24),
        "pdata_size": FieldInfo(0, 11),
    }

    def __init__(self):
        self.temp_cube_size = 0
        self.pdata_size = 0
        super().__init__(0x221)


class ItpLinIntOcfg(Register):
    field_infos = {
        "oact_int_height": FieldInfo(0, 11),
        "oact_int_width": FieldInfo(16, 28),
    }

    def __init__(self):
        self.oact_int_height = 0
        self.oact_int_width = 0
        super().__init__(0x222)


class ItpLinIntCcfg0(Register):
    field_infos = {"act_w_interval": FieldInfo(0, 27)}

    def __init__(self):
        self.act_w_interval = 0
        super().__init__(0x223)


class ItpLinIntCcfg1(Register):
    field_infos = {"act_h_interval": FieldInfo(0, 27)}

    def __init__(self):
        self.act_h_interval = 0
        super().__init__(0x224)


class ItpLinIntStartOffset0(Register):
    field_infos = {
        "signed_bit": FieldInfo(31, 31),
        "w_start_offset": FieldInfo(0, 23),
    }

    def __init__(self):
        self.signed_bit = 0
        self.w_start_offset = 0
        super().__init__(0x225)


class ItpLinIntStartOffset1(Register):
    field_infos = {
        "signed_bit": FieldInfo(31, 31),
        "h_start_offset": FieldInfo(0, 23),
    }

    def __init__(self):
        self.signed_bit = 0
        self.h_start_offset = 0
        super().__init__(0x226)


class ItpLutDcfg0(Register):
    field_infos = {"access_data": FieldInfo(0, 15)}

    def __init__(self):
        self.access_data = 0
        super().__init__(0x230)


class ItpLutDcfg1(Register):
    field_infos = {
        "acs_rptr": FieldInfo(16, 24),
        "acs_wptr": FieldInfo(0, 8),
    }

    def __init__(self):
        self.acs_rptr = 0
        self.acs_wptr = 0
        super().__init__(0x231)


# Each ITP has a copy of the following registers.
class ItpEMulActMcfg(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "act"

    def __init__(self, idx):
        self.base_addr = 0
        super().__init__(0x300 + idx * 0x100, 20 + idx * 4)


class ItpEAluActMcfg(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "act"

    def __init__(self, idx):
        self.base_addr = 0
        super().__init__(0x301 + idx * 0x100, 21 + idx * 4)


class ItpM0AluMcfg(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "param"

    def __init__(self, idx):
        self.base_addr = 0
        super().__init__(0x310 + idx * 0x100, 16 + idx * 16)


class ItpM0MulMcfg(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "param"

    def __init__(self, idx):
        self.base_addr = 0
        super().__init__(0x311 + idx * 0x100, 17 + idx * 16)


class ItpM1AluMcfg(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "param"

    def __init__(self, idx):
        self.base_addr = 0
        super().__init__(0x312 + idx * 0x100, 18 + idx * 16)


class ItpM1MulMcfg(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "param"

    def __init__(self, idx):
        self.base_addr = 0
        super().__init__(0x313 + idx * 0x100, 19 + idx * 16)


class ItpEMulPrmMcfg(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "param"

    def __init__(self, idx):
        self.base_addr = 0
        super().__init__(0x314 + idx * 0x100, 20 + idx * 16)


class ItpEAluPrmMcfg(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "param"

    def __init__(self, idx):
        self.base_addr = 0
        super().__init__(0x315 + idx * 0x100, 21 + idx * 16)


class ItpEMulFpopMcfg(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "param"

    def __init__(self, idx):
        self.base_addr = 0
        super().__init__(0x316 + idx * 0x100, 22 + idx * 16)


class ItpEAluFpopMcfg(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "param"

    def __init__(self, idx):
        self.base_addr = 0
        super().__init__(0x317 + idx * 0x100, 23 + idx * 16)


class ItpOcvtBiasMcfg(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "param"

    def __init__(self, idx):
        self.base_addr = 0
        super().__init__(0x318 + idx * 0x100, 24 + idx * 16)


class ItpOcvtScaleMcfg(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "param"

    def __init__(self, idx):
        self.base_addr = 0
        super().__init__(0x319 + idx * 0x100, 25 + idx * 16)


class ItpOcvtTrshMcfg(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "param"

    def __init__(self, idx):
        self.base_addr = 0
        super().__init__(0x31A + idx * 0x100, 26 + idx * 16)


class ItpOfpshMcfg(Register):
    field_infos = {"base_addr": FieldInfo(0, 31)}  # 128-bit aligned.
    desc_type = "param"

    def __init__(self, idx):
        self.base_addr = 0
        super().__init__(0x31B + idx * 0x100, 27 + idx * 16)


class ItpM0Ccfg(Register):
    field_infos = {
        "alu_byp": FieldInfo(0, 0),
        "alu_op": FieldInfo(1, 2),
        "alu_psrc": FieldInfo(3, 4),
        "alu_xlsh": FieldInfo(5, 9),
        "alu_psrc_prec": FieldInfo(10, 11),
        "alu_psrc_dtype": FieldInfo(12, 12),
        "mul_byp": FieldInfo(13, 13),
        "mul_psrc": FieldInfo(14, 15),
        "mul_trsh": FieldInfo(16, 22),
        "mul_op": FieldInfo(23, 23),
        "mul_psrc_prec": FieldInfo(24, 25),
        "mul_psrc_dtype": FieldInfo(26, 26),
        "relu_byp": FieldInfo(27, 27),
        "byp": FieldInfo(31, 31),
    }

    def __init__(self, idx):
        self.alu_byp = 0
        self.alu_op = 0
        self.alu_psrc = 0
        self.alu_xlsh = 0
        self.alu_psrc_prec = 0
        self.alu_psrc_dtype = 0
        self.mul_byp = 0
        self.mul_psrc = 0
        self.mul_trsh = 0
        self.mul_op = 0
        self.mul_psrc_prec = 0
        self.mul_psrc_dtype = 0
        self.relu_byp = 0
        self.byp = 0
        super().__init__(0x320 + idx * 0x100)


class ItpM0AluPcfg(Register):
    field_infos = {"alu_pval": FieldInfo(0, 31)}

    def __init__(self, idx):
        self.alu_pval = 0
        super().__init__(0x321 + idx * 0x100)


class ItpM0MulPcfg(Register):
    field_infos = {"mul_pval": FieldInfo(0, 31)}

    def __init__(self, idx):
        self.mul_pval = 0
        super().__init__(0x322 + idx * 0x100)


class ItpM1Ccfg(Register):
    field_infos = {
        "alu_byp": FieldInfo(0, 0),
        "alu_op": FieldInfo(1, 2),
        "alu_psrc": FieldInfo(3, 4),
        "alu_xlsh": FieldInfo(5, 9),
        "alu_psrc_prec": FieldInfo(10, 11),
        "alu_psrc_dtype": FieldInfo(12, 12),
        "mul_byp": FieldInfo(13, 13),
        "mul_psrc": FieldInfo(14, 15),
        "mul_trsh": FieldInfo(16, 22),
        "mul_op": FieldInfo(23, 23),
        "mul_psrc_prec": FieldInfo(24, 25),
        "mul_psrc_dtype": FieldInfo(26, 26),
        "relu_byp": FieldInfo(27, 27),
        "byp": FieldInfo(31, 31),
    }

    def __init__(self, idx):
        self.alu_byp = 0
        self.alu_op = 0
        self.alu_psrc = 0
        self.alu_xlsh = 0
        self.alu_psrc_prec = 0
        self.alu_psrc_dtype = 0
        self.mul_byp = 0
        self.mul_psrc = 0
        self.mul_trsh = 0
        self.mul_op = 0
        self.mul_psrc_prec = 0
        self.mul_psrc_dtype = 0
        self.relu_byp = 0
        self.byp = 0
        super().__init__(0x323 + idx * 0x100)


class ItpM1AluPcfg(Register):
    field_infos = {"alu_pval": FieldInfo(0, 31)}

    def __init__(self, idx):
        self.alu_pval = 0
        super().__init__(0x324 + idx * 0x100)


class ItpM1MulPcfg(Register):
    field_infos = {"mul_pval": FieldInfo(0, 31)}

    def __init__(self, idx):
        self.mul_pval = 0
        super().__init__(0x325 + idx * 0x100)


class ItpECcfg(Register):
    field_infos = {
        "mul_byp": FieldInfo(0, 0),
        "mul_psrc": FieldInfo(1, 2),
        "mul_op": FieldInfo(3, 3),
        "mul_trsh": FieldInfo(4, 10),
        "mul_psrc_prec": FieldInfo(11, 12),
        "alu_byp": FieldInfo(13, 13),
        "alu_clamp_en": FieldInfo(14, 14),
        "alu_psrc": FieldInfo(15, 16),
        "alu_op": FieldInfo(17, 18),
        "alu_psrc_prec": FieldInfo(19, 20),
        "e_pf_num": FieldInfo(21, 23),
        "mul_op_square": FieldInfo(24, 24),
        "e_alu_type": FieldInfo(25, 25),
        "e_mul_type": FieldInfo(26, 26),
        "byp": FieldInfo(31, 31),
    }

    def __init__(self, idx):
        self.mul_byp = 0
        self.mul_psrc = 0
        self.mul_op = 0
        self.mul_trsh = 0
        self.mul_psrc_prec = 0
        self.alu_byp = 0
        self.alu_clamp_en = 0
        self.alu_psrc = 0
        self.alu_op = 0
        self.alu_psrc_prec = 0
        self.e_pf_num = 0
        self.mul_op_square = 0
        self.e_alu_type = 0
        self.e_mul_type = 0
        self.byp = 0
        super().__init__(0x326 + idx * 0x100)


class ItpEMulPcfg(Register):
    field_infos = {"mul_pval": FieldInfo(0, 31)}

    def __init__(self, idx):
        self.mul_pval = 0
        super().__init__(0x327 + idx * 0x100)


class ItpEAluPcfg(Register):
    field_infos = {"alu_pval": FieldInfo(0, 31)}

    def __init__(self, idx):
        self.alu_pval = 0
        super().__init__(0x328 + idx * 0x100)


class ItpEAluClampPcfg0(Register):
    field_infos = {"hi_val": FieldInfo(0, 31)}

    def __init__(self, idx):
        self.hi_val = 0x7FFFFFFF
        super().__init__(0x329 + idx * 0x100)


class ItpEAluClampPcfg1(Register):
    field_infos = {"lo_val": FieldInfo(0, 31)}

    def __init__(self, idx):
        self.lo_val = 0
        super().__init__(0x32A + idx * 0x100)


class ItpEMulPcvtCcfg(Register):
    field_infos = {
        "cvt_byp": FieldInfo(0, 0),
        "trsh": FieldInfo(1, 6),
        "dtype": FieldInfo(8, 8),
        "e_mul_fp_psrc": FieldInfo(16, 16),
        "e_mul_fp_en": FieldInfo(17, 18),
    }

    def __init__(self, idx):
        self.cvt_byp = 0
        self.trsh = 0
        self.dtype = 0
        self.e_mul_fp_psrc = 0
        self.e_mul_fp_en = 0
        super().__init__(0x32B + idx * 0x100)


class ItpEMulPcvtPcfg(Register):
    field_infos = {
        "bias": FieldInfo(0, 15),
        "k": FieldInfo(16, 31),
    }

    def __init__(self, idx):
        self.bias = 0
        self.k = 0
        super().__init__(0x32C + idx * 0x100)


class ItpEAluPcvtCcfg(Register):
    field_infos = {
        "cvt_byp": FieldInfo(0, 0),
        "trsh": FieldInfo(1, 6),
        "dtype": FieldInfo(8, 8),
        "e_alu_fp_psrc": FieldInfo(16, 16),
        "e_alu_fp_en": FieldInfo(17, 18),
    }

    def __init__(self, idx):
        self.cvt_byp = 0
        self.trsh = 0
        self.dtype = 0
        self.e_alu_fp_psrc = 0
        self.e_alu_fp_en = 0
        super().__init__(0x32D + idx * 0x100)


class ItpEAluPcvtPcfg(Register):
    field_infos = {
        "bias": FieldInfo(0, 15),
        "k": FieldInfo(16, 31),
    }

    def __init__(self, idx):
        self.bias = 0
        self.k = 0
        super().__init__(0x32E + idx * 0x100)


class ItpEMulFpPcfg(Register):
    field_infos = {
        "fp_scale": FieldInfo(0, 7),
        "fp_sh": FieldInfo(16, 23),
    }

    def __init__(self, idx):
        self.fp_scale = 0
        self.fp_sh = 0
        super().__init__(0x32F + idx * 0x100)


class ItpEAluFpPcfg(Register):
    field_infos = {
        "fp_scale": FieldInfo(0, 7),
        "fp_sh": FieldInfo(16, 23),
    }

    def __init__(self, idx):
        self.fp_scale = 0
        self.fp_sh = 0
        super().__init__(0x330 + idx * 0x100)


class ItpLutPcfg0(Register):
    field_infos = {"bias": FieldInfo(0, 31)}

    def __init__(self, idx):
        self.bias = 0
        super().__init__(0x331 + idx * 0x100)


class ItpLutPcfg1(Register):
    field_infos = {"prsh": FieldInfo(0, 4)}

    def __init__(self, idx):
        self.prsh = 0
        super().__init__(0x332 + idx * 0x100)


class ItpLutOcfg(Register):
    field_infos = {
        "max_slope": FieldInfo(0, 7),
        "min_slope": FieldInfo(16, 23),
        "mode": FieldInfo(24, 24),
        "byp": FieldInfo(25, 25),
    }

    def __init__(self, idx):
        self.max_slope = 0
        self.min_slope = 0
        self.mode = 0
        self.byp = 0
        super().__init__(0x333 + idx * 0x100)


class ItpOcvtCfg(Register):
    field_infos = {
        "trsh": FieldInfo(0, 5),
        "trsh_psrc": FieldInfo(6, 6),
        "scale_psrc": FieldInfo(7, 7),
        "bias_psrc": FieldInfo(8, 8),
        "ofp_psrc": FieldInfo(9, 9),
        "ofp_en": FieldInfo(10, 11),
        "prec": FieldInfo(16, 17),
        "dtype": FieldInfo(18, 18),
    }

    def __init__(self, idx):
        self.trsh = 0
        self.trsh_psrc = 0
        self.scale_psrc = 0
        self.bias_psrc = 0
        self.ofp_psrc = 0
        self.ofp_en = 0
        self.prec = 0
        self.dtype = 0
        super().__init__(0x334 + idx * 0x100)


class ItpOcvtSubCfg(Register):
    field_infos = {"bias": FieldInfo(0, 31)}

    def __init__(self, idx):
        self.bias = 0
        super().__init__(0x335 + idx * 0x100)


class ItpOcvtMulCfg(Register):
    field_infos = {
        "fpsh_num": FieldInfo(16, 23),
        "k": FieldInfo(0, 15),
    }

    def __init__(self, idx):
        self.fpsh_num = 0
        self.k = 0
        super().__init__(0x336 + idx * 0x100)


class ItpOcvtAsymCfg(Register):
    field_infos = {"asym": FieldInfo(0, 31)}

    def __init__(self, idx):
        self.asym = 0
        super().__init__(0x337 + idx * 0x100)


class ItpActCcfg0(Register):
    field_infos = {
        "act_height": FieldInfo(0, 11),
        "act_width": FieldInfo(16, 28),
    }

    def __init__(self, idx):
        self.act_height = 0
        self.act_width = 0
        super().__init__(0x338 + idx * 0x100)


class ItpActCcfg1(Register):
    field_infos = {
        "act_chan": FieldInfo(0, 12),
        "op_mode": FieldInfo(16, 16),
        "itp_byp": FieldInfo(17, 17),
        "in_channel": FieldInfo(18, 18),
    }

    def __init__(self, idx):
        self.act_chan = 0
        self.op_mode = 0
        self.itp_byp = 0
        self.in_channel = 0
        super().__init__(0x339 + idx * 0x100)


class ItpActStepCcfg(Register):
    field_infos = {
        "act_wstep_in": FieldInfo(0, 7),
        "act_hstep_in": FieldInfo(8, 15),
        "act_wstep_len": FieldInfo(16, 23),
        "act_hstep_len": FieldInfo(24, 31),
    }

    def __init__(self, idx):
        self.act_wstep_in = 0
        self.act_hstep_in = 0
        self.act_wstep_len = 0
        self.act_hstep_len = 0
        super().__init__(0x33A + idx * 0x100)


class ItpEMulSurfStride(Register):
    field_infos = {"param_stride": FieldInfo(5, 30)}

    def __init__(self, idx):
        self.param_stride = 0
        super().__init__(0x33B + idx * 0x100)


class ItpEMulTotalStride(Register):
    field_infos = {"total_stride": FieldInfo(3, 31)}

    def __init__(self, idx):
        self.total_stride = 0
        super().__init__(0x33C + idx * 0x100)


class ItpEMulWidthStride(Register):
    field_infos = {"width_stride": FieldInfo(5, 19)}

    def __init__(self, idx):
        self.width_stride = 0
        super().__init__(0x33D + idx * 0x100)


class ItpEMulBatchStride(Register):
    field_infos = {"batch_stride": FieldInfo(3, 31)}

    def __init__(self, idx):
        self.batch_stride = 0
        super().__init__(0x33E + idx * 0x100)


class ItpEAluSurfStride(Register):
    field_infos = {"param_stride": FieldInfo(5, 30)}

    def __init__(self, idx):
        self.param_stride = 0
        super().__init__(0x33F + idx * 0x100)


class ItpEAluTotalStride(Register):
    field_infos = {"total_stride": FieldInfo(3, 31)}

    def __init__(self, idx):
        self.total_stride = 0
        super().__init__(0x340 + idx * 0x100)


class ItpEAluWidthStride(Register):
    field_infos = {"width_stride": FieldInfo(5, 19)}

    def __init__(self, idx):
        self.width_stride = 0
        super().__init__(0x341 + idx * 0x100)


class ItpEAluBatchStride(Register):
    field_infos = {"batch_stride": FieldInfo(3, 31)}

    def __init__(self, idx):
        self.batch_stride = 0
        super().__init__(0x342 + idx * 0x100)


#
# Planar-to-Tensor Processing (PTP) Registers
#
class PtpIactAddr(Register):
    field_infos = {"iact_addr": FieldInfo(3, 31)}
    desc_type = "act"

    def __init__(self):
        self.iact_addr = 0
        super().__init__(0x600, 32)


class PtpCtagAddr(Register):
    field_infos = {"ctag_base_addr": FieldInfo(0, 31)}
    desc_type = "act"

    def __init__(self):
        self.ctag_base_addr = 0
        super().__init__(0x601, 33)


class PtpRoipDespAddr(Register):
    field_infos = {"roip_desp_addr": FieldInfo(4, 31)}
    desc_type = "act"

    def __init__(self):
        self.roip_desp_addr = 0
        super().__init__(0x602, 34)


class PtpPadAddr(Register):
    field_infos = {"pad_addr": FieldInfo(3, 31)}
    desc_type = "param"

    def __init__(self):
        self.pad_addr = 0
        super().__init__(0x610, 64)


class PtpWeightAddr(Register):
    field_infos = {"weight_addr": FieldInfo(3, 31)}
    desc_type = "param"

    def __init__(self):
        self.weight_addr = 0
        super().__init__(0x611, 65)


class PtpBiasAddr(Register):
    field_infos = {"bias_addr": FieldInfo(3, 31)}
    desc_type = "param"

    def __init__(self):
        self.bias_addr = 0
        super().__init__(0x612, 66)


class PtpScaleAddr(Register):
    field_infos = {"scale_addr": FieldInfo(3, 31)}
    desc_type = "param"

    def __init__(self):
        self.scale_addr = 0
        super().__init__(0x613, 67)


class PtpShiftAddr(Register):
    field_infos = {"shift_addr": FieldInfo(3, 31)}
    desc_type = "param"

    def __init__(self):
        self.shift_addr = 0
        super().__init__(0x614, 68)


class PtpIactFpParamAddr(Register):
    field_infos = {"iact_fp_param_addr": FieldInfo(3, 31)}
    desc_type = "param"

    def __init__(self):
        self.iact_fp_param_addr = 0
        super().__init__(0x615, 69)


class PtpOactFpParamAddr(Register):
    field_infos = {"oact_fp_param_addr": FieldInfo(3, 31)}
    desc_type = "param"

    def __init__(self):
        self.oact_fp_param_addr = 0
        super().__init__(0x616, 70)


class PtpMode(Register):
    field_infos = {
        "mode": FieldInfo(0, 2),
        "pad_en": FieldInfo(4, 4),
        "pad_mode": FieldInfo(5, 5),
        "op_fmt": FieldInfo(8, 10),
        "relu_en": FieldInfo(11, 11),
        "batch_num": FieldInfo(12, 13),
        "deconv_en": FieldInfo(14, 14),
        "roi_mode": FieldInfo(16, 18),
        "roi_num": FieldInfo(24, 31),
    }

    def __init__(self):
        self.mode = 0
        self.pad_en = 0
        self.pad_mode = 0
        self.op_fmt = 0
        self.relu_en = 0
        self.batch_num = 0
        self.deconv_en = 0
        self.roi_mode = 0
        self.roi_num = 0
        super().__init__(0x620)


class PtpKernel(Register):
    field_infos = {
        "kernel_w": FieldInfo(0, 2),
        "kernel_h": FieldInfo(3, 5),
        "stride_w": FieldInfo(8, 10),
        "stride_h": FieldInfo(12, 14),
        "dilation": FieldInfo(16, 18),
        "reduction_num": FieldInfo(24, 26),
    }

    def __init__(self):
        self.kernel_w = 0
        self.kernel_h = 0
        self.stride_w = 0
        self.stride_h = 0
        self.dilation = 0
        self.reduction_num = 0
        super().__init__(0x621)


class PtpPadSize(Register):
    field_infos = {
        "pad_right": FieldInfo(0, 4),
        "pad_left": FieldInfo(8, 12),
        "pad_bottom": FieldInfo(16, 20),
        "pad_deconv_y": FieldInfo(21, 21),
        "pad_top": FieldInfo(24, 28),
        "pad_deconv_x": FieldInfo(29, 29),
    }

    def __init__(self):
        self.pad_right = 0
        self.pad_left = 0
        self.pad_bottom = 0
        self.pad_deconv_y = 0
        self.pad_top = 0
        self.pad_deconv_x = 0
        super().__init__(0x622)


class PtpPadValue(Register):
    field_infos = {"pad_value": FieldInfo(0, 15)}

    def __init__(self):
        self.pad_value = 0
        super().__init__(0x623)


class PtpBiasCtrl(Register):
    field_infos = {
        "bias_type": FieldInfo(0, 0),
        "bias_fmt": FieldInfo(1, 1),
        "bias_shift": FieldInfo(8, 12),
    }

    def __init__(self):
        self.bias_type = 0
        self.bias_fmt = 0
        self.bias_shift = 0
        super().__init__(0x624)


class PtpBiasValue(Register):
    field_infos = {"bias_value": FieldInfo(0, 31)}

    def __init__(self):
        self.bias_value = 0
        super().__init__(0x625)


class PtpScaleCtrl(Register):
    field_infos = {
        "scale_type": FieldInfo(0, 0),
        "shift_type": FieldInfo(1, 1),
        "scale_shift": FieldInfo(8, 13),
        "scale_value": FieldInfo(16, 31),
    }

    def __init__(self):
        self.scale_type = 0
        self.shift_type = 0
        self.scale_shift = 0
        self.scale_value = 0
        super().__init__(0x626)


class PtpAsymValue(Register):
    field_infos = {"asym_value": FieldInfo(0, 31)}

    def __init__(self):
        self.asym_value = 0
        super().__init__(0x627)


class PtpFpParam(Register):
    field_infos = {
        "input_fp_scale": FieldInfo(0, 7),
        "input_fp_shift": FieldInfo(8, 15),
        "output_fp_shift": FieldInfo(16, 23),
    }

    def __init__(self):
        self.input_fp_scale = 0
        self.input_fp_shift = 0
        self.output_fp_shift = 0
        super().__init__(0x628)


class PtpWStep(Register):
    field_infos = {
        "w_step_in": FieldInfo(0, 12),
        "w_step_len": FieldInfo(16, 28),
    }

    def __init__(self):
        self.w_step_in = 0
        self.w_step_len = 0
        super().__init__(0x629)


class PtpHStep(Register):
    field_infos = {
        "h_step_in": FieldInfo(0, 11),
        "h_step_len": FieldInfo(16, 27),
    }

    def __init__(self):
        self.h_step_in = 0
        self.h_step_len = 0
        super().__init__(0x62A)


class PtpStepOut(Register):
    field_infos = {
        "w_step_out": FieldInfo(0, 12),
        "h_step_out": FieldInfo(16, 27),
    }

    def __init__(self):
        self.w_step_out = 0
        self.h_step_out = 0
        super().__init__(0x62B)


class PtpCStep(Register):
    field_infos = {"c_step_out": FieldInfo(0, 8)}

    def __init__(self):
        self.c_step_out = 0
        super().__init__(0x62C)


class PtpIactCtrl(Register):
    field_infos = {
        "iact_src": FieldInfo(0, 0),
        "iact_enhanced_sparse": FieldInfo(8, 8),
        "act_zero_skip_en": FieldInfo(9, 9),
        "input_convert": FieldInfo(16, 17),
        "input_fp_param_type": FieldInfo(18, 18),
        "output_convert": FieldInfo(24, 25),
        "output_fp_param_type": FieldInfo(26, 26),
    }

    def __init__(self):
        self.iact_src = 0
        self.iact_enhanced_sparse = 0
        self.act_zero_skip_en = 0
        self.input_convert = 0
        self.input_fp_param_type = 0
        self.output_convert = 0
        self.output_fp_param_type = 0
        super().__init__(0x62D)


class PtpWeightSparse(Register):
    field_infos = {
        "wt_sparse_en": FieldInfo(0, 0),
        "compression_format": FieldInfo(1, 2),
        "wt_zero_skip_en": FieldInfo(3, 3),
        "wt_size": FieldInfo(4, 31),
    }

    def __init__(self):
        self.wt_sparse_en = 0
        self.compression_format = 0
        self.wt_zero_skip_en = 0
        self.wt_size = 0
        super().__init__(0x62E)


class PtpActCCtrl(Register):
    field_infos = {
        "iact_chan": FieldInfo(0, 12),
        "oact_chan": FieldInfo(16, 28),
    }

    def __init__(self):
        self.iact_chan = 0
        self.oact_chan = 0
        super().__init__(0x62F)


class PtpActWCtrl(Register):
    field_infos = {
        "iact_width": FieldInfo(0, 12),
        "oact_width": FieldInfo(16, 28),
    }

    def __init__(self):
        self.iact_width = 0
        self.oact_width = 0
        super().__init__(0x630)


class PtpActHCtrl(Register):
    field_infos = {
        "iact_height": FieldInfo(0, 11),
        "oact_height": FieldInfo(16, 27),
    }

    def __init__(self):
        self.iact_height = 0
        self.oact_height = 0
        super().__init__(0x631)


class PtpPadUnbAddr(Register):
    field_infos = {"pad_unb_addr": FieldInfo(0, 13)}

    def __init__(self):
        self.pad_unb_addr = 0
        super().__init__(0x632)


class PtpWeightUnbAddr(Register):
    field_infos = {"weight_unb_addr": FieldInfo(0, 13)}

    def __init__(self):
        self.weight_unb_addr = 0
        super().__init__(0x633)


class PtpBiasUnbAddr(Register):
    field_infos = {"bias_unb_addr": FieldInfo(0, 13)}

    def __init__(self):
        self.bias_unb_addr = 0
        super().__init__(0x634)


class PtpScaleUnbAddr(Register):
    field_infos = {
        "scale_unb_addr": FieldInfo(0, 13),
        "shift_unb_addr": FieldInfo(16, 29),
    }

    def __init__(self):
        self.scale_unb_addr = 0
        self.shift_unb_addr = 0
        super().__init__(0x635)


class PtpFpParamUnbAddr(Register):
    field_infos = {
        "iact_fp_param_unb_addr": FieldInfo(0, 13),
        "oact_fp_param_unb_addr": FieldInfo(16, 29),
    }

    def __init__(self):
        self.iact_fp_param_unb_addr = 0
        self.oact_fp_param_unb_addr = 0
        super().__init__(0x636)


class PtpRoipUnbAddr(Register):
    field_infos = {"roip_unb_addr": FieldInfo(0, 13)}

    def __init__(self):
        self.roip_unb_addr = 0
        super().__init__(0x637)


class PtpIactWStride(Register):
    field_infos = {"iact_width_stride": FieldInfo(0, 30)}

    def __init__(self):
        self.iact_width_stride = 0
        super().__init__(0x638)


class PtpIactSurfStride(Register):
    field_infos = {"iact_surface_stride": FieldInfo(5, 30)}

    def __init__(self):
        self.iact_surface_stride = 0
        super().__init__(0x639)


class PtpIactTotalStride(Register):
    field_infos = {"iact_total_stride": FieldInfo(3, 31)}

    def __init__(self):
        self.iact_total_stride = 0
        super().__init__(0x63A)


class PtpIactBatchStride(Register):
    field_infos = {"iact_batch_stride": FieldInfo(3, 31)}

    def __init__(self):
        self.iact_batch_stride = 0
        super().__init__(0x63B)


class PtpIactCtagSurfStride(Register):
    field_infos = {"ctag_surface_stride": FieldInfo(0, 25)}

    def __init__(self):
        self.ctag_surface_stride = 0
        super().__init__(0x63C)


class PtpIactCtagBatchStride(Register):
    field_infos = {"ctag_batch_stride": FieldInfo(0, 31)}

    def __init__(self):
        self.ctag_batch_stride = 0
        super().__init__(0x63D)
