# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The X3 specific part of AIFF function unit."""
from ..aiff import RegisterConfigBase
from ..function_unit import FunctionUnit, MultipleUnit
from . import register as reg


class Unb(FunctionUnit):
    """The Unified Buffer unit of AIFF."""

    def __init__(self):
        self.mtp0_base_addr = reg.UnbMtp0BaseAddr()
        self.mtp1_base_addr = reg.UnbMtp1BaseAddr()
        self.itp0_base_addr = reg.UnbItp0BaseAddr()
        self.itp1_base_addr = reg.UnbItp1BaseAddr()
        self.ptp_base_addr = reg.UnbPtpBaseAddr()
        super().__init__()


class Wrb(FunctionUnit):
    """The Write Buffer unit of AIFF."""

    def __init__(self):
        self.region0_oact_addr = reg.WrbRegion0OactAddr()
        self.region1_oact_addr = reg.WrbRegion1OactAddr()
        self.region0_crop0_addr = reg.WrbRegion0Crop0Addr()
        self.region0_crop1_addr = reg.WrbRegion0Crop1Addr()
        self.region1_crop0_addr = reg.WrbRegion1Crop0Addr()
        self.region1_crop1_addr = reg.WrbRegion1Crop1Addr()
        self.mode_ctrl = reg.WrbModeCtrl()
        self.region0_oact_ctrl = reg.WrbRegion0OactCtrl()
        self.region0_oact_l_stride = reg.WrbRegion0OactLStride()
        self.region0_oact_s_stride = reg.WrbRegion0OactSStride()
        self.region0_int_ls_stride = reg.WrbRegion0IntLsStride()
        self.region0_oact_wh_offset = reg.WrbRegion0OactWhOffset()
        self.region1_placeholder = reg.WrbRegion1Placeholder()
        self.region1_oact_ctrl = reg.WrbRegion1OactCtrl()
        self.region1_oact_l_stride = reg.WrbRegion1OactLStride()
        self.region1_oact_s_stride = reg.WrbRegion1OactSStride()
        self.region1_int_ls_stride = reg.WrbRegion1IntLsStride()
        self.region1_oact_wh_offset = reg.WrbRegion1OactWhOffset()
        self.region0_crop0_tl_index = reg.WrbRegion0Crop0TlIndex()
        self.region0_crop1_tl_index = reg.WrbRegion0Crop1TlIndex()
        self.region0_crop0_br_index = reg.WrbRegion0Crop0BrIndex()
        self.region0_crop1_br_index = reg.WrbRegion0Crop1BrIndex()
        self.region0_crop0_l_stride = reg.WrbRegion0Crop0LStride()
        self.region0_crop1_l_stride = reg.WrbRegion0Crop1LStride()
        self.region0_crop0_s_stride = reg.WrbRegion0Crop0SStride()
        self.region0_crop1_s_stride = reg.WrbRegion0Crop1SStride()
        self.region0_crop0_wh_offset = reg.WrbRegion0Crop0WhOffset()
        self.region0_crop1_wh_offset = reg.WrbRegion0Crop1WhOffset()
        self.region1_crop0_tl_index = reg.WrbRegion1Crop0TlIndex()
        self.region1_crop1_tl_index = reg.WrbRegion1Crop1TlIndex()
        self.region1_crop0_br_index = reg.WrbRegion1Crop0BrIndex()
        self.region1_crop1_br_index = reg.WrbRegion1Crop1BrIndex()
        self.region1_crop0_l_stride = reg.WrbRegion1Crop0LStride()
        self.region1_crop1_l_stride = reg.WrbRegion1Crop1LStride()
        self.region1_crop0_s_stride = reg.WrbRegion1Crop0SStride()
        self.region1_crop1_s_stride = reg.WrbRegion1Crop1SStride()
        self.region1_crop0_wh_offset = reg.WrbRegion1Crop0WhOffset()
        self.region1_crop1_wh_offset = reg.WrbRegion1Crop1WhOffset()
        super().__init__()


class MtpUnit(FunctionUnit):
    """The work unit of Matrix-to-Tensor Processing."""

    def __init__(self, idx):
        self.iact_addr = reg.MtpIactAddr(idx)
        self.batch_matmul_iact_addr = reg.MtpBatchMatmulIactAddr(idx)
        self.unb_addr = reg.MtpUnbAddr(idx)
        self.iact_gr0_addr = reg.MtpIactGr0Addr(idx)
        self.iact_gr1_addr = reg.MtpIactGr1Addr(idx)
        self.iact_gr2_addr = reg.MtpIactGr2Addr(idx)
        self.iact_gr3_addr = reg.MtpIactGr3Addr(idx)
        self.weight_addr0 = reg.MtpWeightAddr0(idx)
        self.pad_addr = reg.MtpPadAddr(idx)
        self.iact_fp_param_addr0 = reg.MtpIactFpParamAddr0(idx)
        self.iact_fp_param_addr1 = reg.MtpIactFpParamAddr1(idx)
        self.weight_addr1 = reg.MtpWeightAddr1(idx)
        self.wt_bias_addr = reg.MtpWtBiasAddr(idx)
        self.ctrl = reg.MtpCtrl(idx)
        self.be_ctrl = reg.MtpBeCtrl(idx)
        self.unb_ctrl = reg.MtpUnbCtrl(idx)
        self.kernel_ctrl = reg.MtpKernelCtrl(idx)
        self.weight_sparse = reg.MtpWeightSparse(idx)
        self.weight_size = reg.MtpWeightSize(idx)
        self.w_step = reg.MtpWStep(idx)
        self.h_step = reg.MtpHStep(idx)
        self.conv3d_act_stride = reg.MtpConv3dActStride(idx)
        self.deconv_pad = reg.MtpDeconvPad(idx)
        self.iact_ctrl = reg.MtpIactCtrl(idx)
        self.pad_size = reg.MtpPadSize(idx)
        self.pad_value = reg.MtpPadValue(idx)
        self.fp_param = reg.MtpFpParam(idx)
        self.iact_w_stride = reg.MtpIactWStride(idx)
        self.iact_s_stride = reg.MtpIactSStride(idx)
        self.iact_total_stride = reg.MtpIactTotalStride(idx)
        self.act_c_ctrl = reg.MtpActCCtrl(idx)
        self.act_w_ctrl = reg.MtpActWCtrl(idx)
        self.act_h_ctrl = reg.MtpActHCtrl(idx)
        self.wt_compress0 = reg.MtpWtCompress0(idx)
        self.batch_matmul0 = reg.MtpBatchMatmul0(idx)
        self.batch_matmul1 = reg.MtpBatchMatmul1(idx)
        self.batch_matmul2 = reg.MtpBatchMatmul2(idx)
        self.conv3d_ctrl = reg.MtpConv3dCtrl(idx)
        self.conv3d_wt_stride = reg.MtpConv3dWtStride(idx)
        self.iact_bias_ctrl = reg.MtpIactBiasCtrl(idx)
        self.wt_bias_ctrl = reg.MtpWtBiasCtrl(idx)
        self.wt_compress1 = reg.MtpWtCompress1(idx)
        self.iact_tl_index = reg.MtpIactTlIndex(idx)
        self.iact_br_index = reg.MtpIactBrIndex(idx)
        self.iact_w_stride_gr0 = reg.MtpIactWStrideGr0(idx)
        self.iact_w_stride_gr1 = reg.MtpIactWStrideGr1(idx)
        self.iact_w_stride_gr2 = reg.MtpIactWStrideGr2(idx)
        self.iact_w_stride_gr3 = reg.MtpIactWStrideGr3(idx)
        self.iact_s_stride_gr0 = reg.MtpIactSStrideGr0(idx)
        self.iact_s_stride_gr1 = reg.MtpIactSStrideGr1(idx)
        self.iact_s_stride_gr2 = reg.MtpIactSStrideGr2(idx)
        self.iact_s_stride_gr3 = reg.MtpIactSStrideGr3(idx)
        self.iact_t_stride_gr0 = reg.MtpIactTStrideGr0(idx)
        self.iact_t_stride_gr1 = reg.MtpIactTStrideGr1(idx)
        self.iact_t_stride_gr2 = reg.MtpIactTStrideGr2(idx)
        self.iact_t_stride_gr3 = reg.MtpIactTStrideGr3(idx)
        self.wdc0_lut0 = reg.MtpWdc0Lut0(idx)
        self.wdc0_lut1 = reg.MtpWdc0Lut1(idx)
        self.wdc1_lut0 = reg.MtpWdc1Lut0(idx)
        self.wdc1_lut1 = reg.MtpWdc1Lut1(idx)
        super().__init__()


class Mtp(MultipleUnit):
    """The Matrix-to-Tensor Processing unit of AIFF."""

    def __init__(self):
        self.twin_ctrl = reg.MtpTwinCtrl()
        super().__init__()
        self._units = (MtpUnit(0), MtpUnit(1))

    def get_ctrl_addr2reg(self, reg_cfg):
        ret = super().get_ctrl_addr2reg(reg_cfg)
        if not self.twin_ctrl.mtp_0_disen:
            ret.update(self._units[0].get_ctrl_addr2reg(reg_cfg))
        if not self.twin_ctrl.mtp_1_disen:
            ret.update(self._units[1].get_ctrl_addr2reg(reg_cfg))
        return ret


class ItpUnit(FunctionUnit):
    """The work unit of Intra-Tensor and Inter-Tensor Processing."""

    def __init__(self, idx):
        self.e_mul_act_mcfg = reg.ItpEMulActMcfg(idx)
        self.e_alu_act_mcfg = reg.ItpEAluActMcfg(idx)
        self.e_mul_grt_mcfg = reg.ItpEMulGrtMcfg(idx)
        self.e_mul_grl_mcfg = reg.ItpEMulGrlMcfg(idx)
        self.e_alu_grt_mcfg = reg.ItpEAluGrtMcfg(idx)
        self.e_alu_grl_mcfg = reg.ItpEAluGrlMcfg(idx)
        self.m0_alu_mcfg = reg.ItpM0AluMcfg(idx)
        self.m0_mul_mcfg = reg.ItpM0MulMcfg(idx)
        self.m1_alu_mcfg = reg.ItpM1AluMcfg(idx)
        self.m1_mul_mcfg = reg.ItpM1MulMcfg(idx)
        self.e_mul_prm_mcfg = reg.ItpEMulPrmMcfg(idx)
        self.e_alu_prm_mcfg = reg.ItpEAluPrmMcfg(idx)
        self.e_mul_fpop_mcfg = reg.ItpEMulFpopMcfg(idx)
        self.e_alu_fpop_mcfg = reg.ItpEAluFpopMcfg(idx)
        self.ocvt_bias_mcfg = reg.ItpOcvtBiasMcfg(idx)
        self.ocvt_scale_mcfg = reg.ItpOcvtScaleMcfg(idx)
        self.ocvt_trsh_mcfg = reg.ItpOcvtTrshMcfg(idx)
        self.ofpsh_mcfg = reg.ItpOfpshMcfg(idx)
        self.m0_ccfg = reg.ItpM0Ccfg(idx)
        self.m0_alu_pcfg = reg.ItpM0AluPcfg(idx)
        self.m0_mul_pcfg = reg.ItpM0MulPcfg(idx)
        self.m1_ccfg = reg.ItpM1Ccfg(idx)
        self.m1_alu_pcfg = reg.ItpM1AluPcfg(idx)
        self.m1_mul_pcfg = reg.ItpM1MulPcfg(idx)
        self.e_ccfg = reg.ItpECcfg(idx)
        self.e_ccfg1 = reg.ItpECcfg1(idx)
        self.e_mul_pcfg = reg.ItpEMulPcfg(idx)
        self.e_alu_pcfg = reg.ItpEAluPcfg(idx)
        self.e_alu_clamp_pcfg0 = reg.ItpEAluClampPcfg0(idx)
        self.e_alu_clamp_pcfg1 = reg.ItpEAluClampPcfg1(idx)
        self.e_mul_pcvt_ccfg = reg.ItpEMulPcvtCcfg(idx)
        self.e_mul_pcvt_pcfg = reg.ItpEMulPcvtPcfg(idx)
        self.e_alu_pcvt_ccfg = reg.ItpEAluPcvtCcfg(idx)
        self.e_alu_pcvt_pcfg = reg.ItpEAluPcvtPcfg(idx)
        self.e_mul_fp_pcfg = reg.ItpEMulFpPcfg(idx)
        self.e_alu_fp_pcfg = reg.ItpEAluFpPcfg(idx)
        self.lut_ocfg = reg.ItpLutOcfg(idx)
        self.lut_pcfg0 = reg.ItpLutPcfg0(idx)
        self.lut_pcfg1 = reg.ItpLutPcfg1(idx)
        self.ocvt_cfg = reg.ItpOcvtCfg(idx)
        self.ocvt_sub_cfg = reg.ItpOcvtSubCfg(idx)
        self.ocvt_mul_cfg = reg.ItpOcvtMulCfg(idx)
        self.ocvt_asym_cfg = reg.ItpOcvtAsymCfg(idx)
        self.act_ccfg0 = reg.ItpActCcfg0(idx)
        self.act_ccfg1 = reg.ItpActCcfg1(idx)
        self.act_step_ccfg = reg.ItpActStepCcfg(idx)
        self.e_mul_surf_stride = reg.ItpEMulSurfStride(idx)
        self.e_mul_total_stride = reg.ItpEMulTotalStride(idx)
        self.e_mul_width_stride = reg.ItpEMulWidthStride(idx)
        self.e_alu_surf_stride = reg.ItpEAluSurfStride(idx)
        self.e_alu_total_stride = reg.ItpEAluTotalStride(idx)
        self.e_alu_width_stride = reg.ItpEAluWidthStride(idx)
        self.e_mul_grt_surf_stride = reg.ItpEMulGrtSurfStride(idx)
        self.e_mul_grt_total_stride = reg.ItpEMulGrtTotalStride(idx)
        self.e_mul_grt_width_stride = reg.ItpEMulGrtWidthStride(idx)
        self.e_mul_grl_surf_stride = reg.ItpEMulGrlSurfStride(idx)
        self.e_mul_grl_total_stride = reg.ItpEMulGrlTotalStride(idx)
        self.e_mul_grl_width_stride = reg.ItpEMulGrlWidthStride(idx)
        self.e_alu_grt_surf_stride = reg.ItpEAluGrtSurfStride(idx)
        self.e_alu_grt_total_stride = reg.ItpEAluGrtTotalStride(idx)
        self.e_alu_grt_width_stride = reg.ItpEAluGrtWidthStride(idx)
        self.e_alu_grl_surf_stride = reg.ItpEAluGrlSurfStride(idx)
        self.e_alu_grl_total_stride = reg.ItpEAluGrlTotalStride(idx)
        self.e_alu_grl_width_stride = reg.ItpEAluGrlWidthStride(idx)
        self.e_mul_tl_index = reg.ItpEMulTlIndex(idx)
        self.e_alu_tl_index = reg.ItpEAluTlIndex(idx)
        self.step_num_l = reg.ItpStepNumL(idx)
        self.step_num_h = reg.ItpStepNumH(idx)
        super().__init__()


class Itp(MultipleUnit):
    """The Intra-Tensor and Inter-Tensor Processing unit of AIFF."""

    def __init__(self):
        self.rdc_int_dst_mcfg0 = reg.ItpRdcIntDstMcfg0()
        self.rdc_dst_mcfg1 = reg.ItpRdcDstMcfg1()
        self.intp_mcfg = reg.ItpIntpMcfg()
        self.int_dst_mcfg = reg.Itp1IntDstMcfg()
        self.ipt1_intp_mcfg = reg.Itp1IntpMcfg()
        self.lut_mcfg0 = reg.ItpLutMcfg0()
        self.lut_mcfg1 = reg.ItpLutMcfg1()
        self.mode_ccfg = reg.ItpModeCcfg()
        self.unb_addr_mcfg = reg.ItpUnbAddrMcfg()
        self.lin_int_ocfg = reg.ItpLinIntOcfg()
        self.lin_int_ccfg0 = reg.ItpLinIntCcfg0()
        self.lin_int_ccfg1 = reg.ItpLinIntCcfg1()
        self.lin_int_start_offset0 = reg.ItpLinIntStartOffset0()
        self.lin_int_start_offset1 = reg.ItpLinIntStartOffset1()
        self.itp1_lin_int_ocfg = reg.Itp1LinIntOcfg()
        self.itp1_lin_int_ccfg0 = reg.Itp1LinIntCcfg0()
        self.itp1_lin_int_ccfg1 = reg.Itp1LinIntCcfg1()
        self.itp1_lin_int_start_offset0 = reg.Itp1LinIntStartOffset0()
        self.itp1_lin_int_start_offset1 = reg.Itp1LinIntStartOffset1()
        self.sfu_ccfg0 = reg.ItpSfuCcfg0()
        self.sfu_pcfg0 = reg.ItpSfuPcfg0()
        self.sfu_ccfg1 = reg.ItpSfuCcfg1()
        self.sfu_pcfg1 = reg.ItpSfuPcfg1()
        super().__init__()
        self._units = (ItpUnit(0), ItpUnit(1), ItpUnit(2), ItpUnit(3))

    def get_ctrl_addr2reg(self, reg_cfg):
        ret = super().get_ctrl_addr2reg(reg_cfg)

        units_addr2reg = [x.get_ctrl_addr2reg(reg_cfg) for x in self._units]
        twin_ctrl = reg_cfg.mtp.twin_ctrl
        is_mtp_disable = twin_ctrl.mtp_0_disen and twin_ctrl.mtp_1_disen
        is_single = (not twin_ctrl.mtp_0_disen) and twin_ctrl.mtp_1_disen
        is_interp = self.mode_ccfg.lin_int_en
        is_loop = self.mode_ccfg.loop_en
        is_independent_with_merge = twin_ctrl.twin_op_mode == 0 and twin_ctrl.twin_output == 2

        if is_mtp_disable:
            if is_interp:
                ret.update(units_addr2reg[0])
                ret.update(units_addr2reg[1])
            return ret

        ret.update(units_addr2reg[0])
        if is_single and is_loop:
            ret.update(units_addr2reg[1])
        elif is_independent_with_merge and is_loop:
            ret.update(units_addr2reg[1])
            ret.update(units_addr2reg[3])
        elif is_loop:
            ret.update(units_addr2reg[1])
            ret.update(units_addr2reg[2])
            ret.update(units_addr2reg[3])
        elif not is_single and not is_interp:
            ret.update(units_addr2reg[1])
        return ret


class Ptp(FunctionUnit):
    """The Planar-to-Tensor Processing unit of AIFF."""

    def __init__(self):
        self.iact_addr = reg.PtpIactAddr()
        self.roip_desp_addr = reg.PtpRoipDespAddr()
        self.iact_gr0_addr = reg.PtpIactGr0Addr()
        self.iact_gr1_addr = reg.PtpIactGr1Addr()
        self.iact_gr2_addr = reg.PtpIactGr2Addr()
        self.iact_gr3_addr = reg.PtpIactGr3Addr()
        self.pad_addr = reg.PtpPadAddr()
        self.weight_addr = reg.PtpWeightAddr()
        self.bias_addr = reg.PtpBiasAddr()
        self.scale_addr = reg.PtpScaleAddr()
        self.shift_addr = reg.PtpShiftAddr()
        self.iact_fp_param_addr = reg.PtpIactFpParamAddr()
        self.oact_fp_param_addr = reg.PtpOactFpParamAddr()
        self.mode = reg.PtpMode()
        self.kernel = reg.PtpKernel()
        self.pad_size = reg.PtpPadSize()
        self.pad_value = reg.PtpPadValue()
        self.bias_ctrl = reg.PtpBiasCtrl()
        self.bias_value = reg.PtpBiasValue()
        self.scale_ctrl = reg.PtpScaleCtrl()
        self.asym_value = reg.PtpAsymValue()
        self.fp_param = reg.PtpFpParam()
        self.w_step = reg.PtpWStep()
        self.h_step = reg.PtpHStep()
        self.step_out = reg.PtpStepOut()
        self.c_step = reg.PtpCStep()
        self.iact_ctrl = reg.PtpIactCtrl()
        self.weight_sparse = reg.PtpWeightSparse()
        self.act_c_ctrl = reg.PtpActCCtrl()
        self.act_w_ctrl = reg.PtpActWCtrl()
        self.act_h_ctrl = reg.PtpActHCtrl()
        self.pad_unb_addr = reg.PtpPadUnbAddr()
        self.weight_unb_addr = reg.PtpWeightUnbAddr()
        self.bias_unb_addr = reg.PtpBiasUnbAddr()
        self.scale_unb_addr = reg.PtpScaleUnbAddr()
        self.fp_param_unb_addr = reg.PtpFpParamUnbAddr()
        self.roip_unb_addr = reg.PtpRoipUnbAddr()
        self.iact_w_stride = reg.PtpIactWStride()
        self.iact_surf_stride = reg.PtpIactSurfStride()
        self.iact_total_stride = reg.PtpIactTotalStride()
        self.iact_depth_stride = reg.PtpIactDepthStride()
        self.iact_tl_index = reg.PtpIactTlIndex()
        self.iact_br_index = reg.PtpIactBrIndex()
        self.iact_gr0_w_stride = reg.PtpIactGr0WStride()
        self.iact_gr1_w_stride = reg.PtpIactGr1WStride()
        self.iact_gr2_w_stride = reg.PtpIactGr2WStride()
        self.iact_gr3_w_stride = reg.PtpIactGr3WStride()
        self.iact_gr0_surf_stride = reg.PtpIactGr0SurfStride()
        self.iact_gr1_surf_stride = reg.PtpIactGr1SurfStride()
        self.iact_gr2_surf_stride = reg.PtpIactGr2SurfStride()
        self.iact_gr3_surf_stride = reg.PtpIactGr3SurfStride()
        self.iact_gr0_total_stride = reg.PtpIactGr0TotalStride()
        self.iact_gr1_total_stride = reg.PtpIactGr1TotalStride()
        self.iact_gr2_total_stride = reg.PtpIactGr2TotalStride()
        self.iact_gr3_total_stride = reg.PtpIactGr3TotalStride()
        super().__init__()


class RegisterConfig(RegisterConfigBase):
    """Combination of all AIFF function units."""

    def __init__(self):
        super().__init__()
        self.unb = Unb()
        self.wrb = Wrb()
        self.mtp = Mtp()
        self.itp = Itp()
        self.ptp = Ptp()
