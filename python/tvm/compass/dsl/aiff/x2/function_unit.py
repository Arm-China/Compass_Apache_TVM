# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The X2 specific part of AIFF function unit."""
from ....logger import WARN
from ..aiff import RegisterConfigBase
from ..function_unit import FunctionUnit, MultipleUnit
from . import register as reg


class Sys(FunctionUnit):
    """The System unit of AIFF."""

    def __init__(self):
        self.ctrl = reg.AiffCtrl()
        self.status = reg.AiffStatus()
        super().__init__()


class Unb(FunctionUnit):
    """The Unified Buffer unit of AIFF."""

    def __init__(self):
        self.mtp0_base_addr = reg.UnbMtp0BaseAddr()
        self.mtp1_base_addr = reg.UnbMtp1BaseAddr()
        self.itp_base_addr = reg.UnbItpBaseAddr()
        self.ptp_base_addr = reg.UnbPtpBaseAddr()
        super().__init__()


class Wrb(FunctionUnit):
    """The Write Buffer unit of AIFF."""

    def __init__(self):
        self.region0_oact_addr = reg.WrbRegion0OactAddr()
        self.region0_oact_ctag_addr = reg.WrbRegion0OactCtagAddr()
        self.region1_oact_addr = reg.WrbRegion1OactAddr()
        self.region1_oact_ctag_addr = reg.WrbRegion1OactCtagAddr()
        self.mode_ctrl = reg.WrbModeCtrl()
        self.region0_oact_ctrl = reg.WrbRegion0OactCtrl()
        self.region0_oact_l_stride = reg.WrbRegion0OactLStride()
        self.region0_oact_s_stride = reg.WrbRegion0OactSStride()
        self.region0_int_ls_stride = reg.WrbRegion0IntLsStride()
        self.region0_octag_s_stride = reg.WrbRegion0OctagSStride()
        self.region0_oact_batch_stride = reg.WrbRegion0OactBatchStride()
        self.region0_octag_batch_stride = reg.WrbRegion0OctagBatchStride()
        self.region0_oact_wh_offset = reg.WrbRegion0OactWhOffset()
        self.region1_placeholder = reg.WrbRegion1Placeholder()
        self.region1_oact_ctrl = reg.WrbRegion1OactCtrl()
        self.region1_oact_l_stride = reg.WrbRegion1OactLStride()
        self.region1_oact_s_stride = reg.WrbRegion1OactSStride()
        self.region1_int_ls_stride = reg.WrbRegion1IntLsStride()
        self.region1_octag_s_stride = reg.WrbRegion1OctagSStride()
        self.region1_oact_batch_stride = reg.WrbRegion1OactBatchStride()
        self.region1_octag_batch_stride = reg.WrbRegion1OctagBatchStride()
        self.region1_oact_wh_offset = reg.WrbRegion1OactWhOffset()
        super().__init__()


class MtpUnit(FunctionUnit):
    """The work unit of Matrix-to-Tensor Processing."""

    def __init__(self, idx):
        self.iact_addr = reg.MtpIactAddr(idx)
        self.batch_matmul_iact_addr = reg.MtpBatchMatmulIactAddr(idx)
        self.iact_ctag_addr = reg.MtpIactCtagAddr(idx)
        self.unb_addr = reg.MtpUnbAddr(idx)
        self.weight_addr = reg.MtpWeightAddr(idx)
        self.pad_addr = reg.MtpPadAddr(idx)
        self.iact_fp_param_addr0 = reg.MtpIactFpParamAddr0(idx)
        self.iact_fp_param_addr1 = reg.MtpIactFpParamAddr1(idx)
        self.ctrl = reg.MtpCtrl(idx)
        self.be_ctrl = reg.MtpBeCtrl(idx)
        self.unb_ctrl = reg.MtpUnbCtrl(idx)
        self.kernel_ctrl = reg.MtpKernelCtrl(idx)
        self.weight_sparse = reg.MtpWeightSparse(idx)
        self.weight_size = reg.MtpWeightSize(idx)
        self.w_step = reg.MtpWStep(idx)
        self.h_step = reg.MtpHStep(idx)
        self.batch_stride = reg.MtpBatchStride(idx)
        self.deconv_pad = reg.MtpDeconvPad(idx)
        self.iact_ctag_surf_stride = reg.MtpIactCtagSurfStride(idx)
        self.iact_ctag_batch_stride = reg.MtpIactCtagBatchStride(idx)
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
        self.wt_compress = reg.MtpWtCompress(idx)
        self.batch_matmul0 = reg.MtpBatchMatmul0(idx)
        self.batch_matmul1 = reg.MtpBatchMatmul1(idx)
        self.batch_matmul2 = reg.MtpBatchMatmul2(idx)
        self.conv3d_ctrl = reg.MtpConv3dCtrl(idx)
        self.conv3d_wt_stride = reg.MtpConv3dWtStride(idx)
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
        self.lut_pcfg0 = reg.ItpLutPcfg0(idx)
        self.lut_pcfg1 = reg.ItpLutPcfg1(idx)
        self.lut_ocfg = reg.ItpLutOcfg(idx)
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
        self.e_mul_batch_stride = reg.ItpEMulBatchStride(idx)
        self.e_alu_surf_stride = reg.ItpEAluSurfStride(idx)
        self.e_alu_total_stride = reg.ItpEAluTotalStride(idx)
        self.e_alu_width_stride = reg.ItpEAluWidthStride(idx)
        self.e_alu_batch_stride = reg.ItpEAluBatchStride(idx)
        super().__init__()


class Itp(MultipleUnit):
    """The Intra-Tensor and Inter-Tensor Processing unit of AIFF."""

    def __init__(self):
        self.acc_int_dst_mcfg0 = reg.ItpAccIntDstMcfg0()
        self.acc_int_dst_mcfg1 = reg.ItpAccIntDstMcfg1()
        self.intp_mcfg = reg.ItpIntpMcfg()
        self.lut_mcfg = reg.ItpLutMcfg()
        self.mode_ccfg = reg.ItpModeCcfg()
        self.unb_addr_mcfg = reg.ItpUnbAddrMcfg()
        self.lin_int_ocfg = reg.ItpLinIntOcfg()
        self.lin_int_ccfg0 = reg.ItpLinIntCcfg0()
        self.lin_int_ccfg1 = reg.ItpLinIntCcfg1()
        self.lin_int_start_offset0 = reg.ItpLinIntStartOffset0()
        self.lin_int_start_offset1 = reg.ItpLinIntStartOffset1()
        self.lut_dcfg0 = reg.ItpLutDcfg0()
        self.lut_dcfg1 = reg.ItpLutDcfg1()
        super().__init__()
        self._units = (ItpUnit(0), ItpUnit(1), ItpUnit(2))

    def get_ctrl_addr2reg(self, reg_cfg):
        ret = super().get_ctrl_addr2reg(reg_cfg)

        # Remove the deprecated registers and report warning if needed.
        del ret[self.lut_dcfg0.addr]
        del ret[self.lut_dcfg1.addr]
        if self.lut_dcfg0.all_fields != 0:
            WARN(f'Deprecated: Register "{self.lut_dcfg0.name}" won\'t be added into descriptor!')
        if self.lut_dcfg1.all_fields != 0:
            WARN(f'Deprecated: Register "{self.lut_dcfg1.name}" won\'t be added into descriptor!')

        units_addr2reg = [x.get_ctrl_addr2reg(reg_cfg) for x in self._units]
        twin_ctrl = reg_cfg.mtp.twin_ctrl
        is_mtp_disable = twin_ctrl.mtp_0_disen and twin_ctrl.mtp_1_disen
        is_single = (not twin_ctrl.mtp_0_disen) and twin_ctrl.mtp_1_disen
        is_interp = self.mode_ccfg.lin_int_en
        is_loop = self.mode_ccfg.loop_en

        if is_mtp_disable:
            if is_interp:
                ret.update(units_addr2reg[0])
            return ret

        ret.update(units_addr2reg[0])
        if is_single and is_loop:
            ret.update(units_addr2reg[2])
        elif is_loop:
            ret.update(units_addr2reg[1])
            ret.update(units_addr2reg[2])
        elif not is_single and not is_interp:
            ret.update(units_addr2reg[1])
        return ret


class Ptp(FunctionUnit):
    """The Planar-to-Tensor Processing unit of AIFF."""

    def __init__(self):
        self.iact_addr = reg.PtpIactAddr()
        self.ctag_addr = reg.PtpCtagAddr()
        self.roip_desp_addr = reg.PtpRoipDespAddr()
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
        self.iact_batch_stride = reg.PtpIactBatchStride()
        self.iact_ctag_surf_stride = reg.PtpIactCtagSurfStride()
        self.iact_ctag_batch_stride = reg.PtpIactCtagBatchStride()
        super().__init__()


class RegisterConfig(RegisterConfigBase):
    """Combination of all AIFF function units."""

    def __init__(self):
        super().__init__()
        self.sys = Sys()
        self.unb = Unb()
        self.wrb = Wrb()
        self.mtp = Mtp()
        self.itp = Itp()
        self.ptp = Ptp()

    def get_register(self, addr):
        ret = self.sys.get_register(addr)
        if ret is not None:
            return ret
        return super().get_register(addr)

    def _get_ctrl_addr2reg(self):
        ret = self.sys.get_ctrl_addr2reg(self)
        ret.update(super()._get_ctrl_addr2reg())
        return ret
