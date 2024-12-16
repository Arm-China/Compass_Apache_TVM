# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import rand


def get_maxpool_gt(inp):
    # assume input layout is NCHWC32
    inp = np.transpose(inp, axes=[0, 2, 3, 1, 4])
    inp = inp.reshape(1, 112, 112, 64)

    pad_inp = np.zeros((1, 113, 113, 64), dtype=np.uint8)
    pad_inp[0, :112, :112, :] = inp
    gt_out = np.empty((1, 56, 56, 64), dtype=np.uint8)
    for i in range(56):
        for j in range(56):
            for c in range(64):
                val = np.max(pad_inp[0, 2 * i : 2 * i + 3, 2 * j : 2 * j + 3, c])
                gt_out[0, i, j, c] = val
    return gt_out


def gen_aiff_maxpool():

    """
    INPUT
    input shape: [1,2,112,112,32]
    input layout: NCHWC32
    input dtype: uint8

    OUTPUT
    output shape: [1,56,56,64]
    output layout: NHWC
    output dtype: uint8

    MaxPooling
    dilation_x: 1
    dilation_y: 1
    kernel_x: 3
    kernel_y: 3
    pad_bottom: 1
    pad_left: 0
    pad_right: 1
    pad_top: 0
    stride_x: 2
    stride_y: 2

    AIFF strategy:
    n: 1, d: 1, h: 112, w: 112, c: 64, c0: 2, c1: 32, c2: 0, od: 1, dtype: uint8, is_need_wrb: 1
    w_step_in: 112, w_step_len: 113, w_step_out: 56, h_step_in: 56, h_step_len: 57, h_step_out: 28,
    w_next_overlap: 0, h_next_overlap: 0, c_step_out: 32
    w_split_num: 1, h_split_num: 2, c_split_num: 2

    split c and h to make sure the otuput size less than 64k (wrb size)
    """

    @S.prim_func
    def test_aiff_max_pooling(inp: S.ptr("uint8", "global"), out: S.ptr("uint8", "global")):
        # ctrl_desc is variable length, offline calculate is 13x8
        ctrl_desc = S.alloc_buffer((13), "uint32x8", "private")
        # param_desc act_desc is fixed length.
        param_desc = S.alloc_buffer((72), "uint32", "private")
        act_desc = S.alloc_buffer((48), "uint32", "private")

        tid = S.get_local_id()

        for i in range(72):
            param_desc[i] = 0

        for i in range(48):
            act_desc[i] = 0

        S.barrier()

        if tid == 0:
            # fill ptp_iact_addr
            # ptp_iact_addr register address is 0x600, and ptp start addr is 0x600, so offset is 0
            # and the ptp desc offset is 32(X2 target), thus fill the 32 slot.
            act_desc[32] = S.reinterpret(inp, "uint32")

            # fill wrb_region1_oact_addr
            # wrb region1 to ddr
            # wrb_region1_oact_addr addr is 0x32, and wrb start addr is 0x30, so offset is 2
            # and the wrb desc offset is 40(X2 target), thus fill the 42 slot.
            act_desc[42] = S.reinterpret(out, "uint32")

            val = S.uint32x8(0)
            """
            // descriptor header:
            // 0. EOF/NOE
            // 1. next desc offset addr
            // 2. next desc offset length
            // 3. current desc offset length
            // 4. loop config
            //    (1) [16] param_desc_addr_inc_mode
            //    (2) [15:0] loop_num
            // 5-7. reserved
            """
            # 0. 0x81 is EOF
            val[0] = 0x81
            # 1. no next desc
            val[1] = 0
            # 2. no next desc len
            val[2] = 0
            # 3. cur desc len is 13x8, fill 13
            val[3] = 13
            # 4. no loop
            val[4] = 0
            ctrl_desc[0] = val

            """
            // descriptor format:
            // 1. csr base addr and csr len:
            //    (1) [21:16] csr length
            //    (2) [10:0]  csr addr
            // 2. multiple csr register value
            // 3. reserved to make sure it's 256-bit aligned
            """

            """
            fill AIFF_CTRL 0x0
            1 register
            head value is (1 << 16) + 0
            Z5_AIFF_CTRL fill 0 as set act_wt_route to port 0
            """
            val = S.uint32x8(0)
            val[0] = 0x10000
            ctrl_desc[1] = val

            """
            fill MTP_TWIN_CTRL 0x170
            1 register
            head value (1 << 16) + 0x170
            MTP_TWIN_CTRL fill 3 as disable MTP0 and MTP1
            """
            val = S.uint32x8(0)
            val[0] = 0x00010170
            val[1] = 3
            ctrl_desc[2] = val

            """
            fill ITP_MODE_CCFG(0x220) to ITP_LIN_INT_START_OFFSET1(0x226)
            7 register
            as itp not used, set them all to zero
            """
            val = S.uint32x8(0)
            val[0] = 0x00070220
            ctrl_desc[3] = val

            """
            fill 30 reigsters (start addr 0x620)
            PTP_MODE 0x00000012, means PTP_mode max pooling, pad_enable and pad_mode is per-layer
                    input is 8bit unsigned and output is unsigned
                    disable relu
                    batch_num is 1
            PTP_KERNEL 0x0000221b means pooling kernel_w 3 kernle_h 3, stride_w 2 stride_h 2
            PTP_PAD_SIZE 0x00010001 means pad_bottom 1 pad_right 1
            PTP_PAD_VALUE 0
            PTP_BIAS_CTRL 0
            PTP_BIAS_VALUE 0
            PTP_SCALE_CTRL 0
            PTP_ASYM_VALUE 0
            PTP_FP_PARAM 0
            PTP_W_STEP 0x00710070 means w_step_in 112, w_step_len 113(pad 1, 112 + 1)
            PTP_H_STEP 0x00390038 means h_step_in: 56, h_step_len: 57
            PTP_STEP_OUT 0x001c0038 means w_step_out 56, h_step_out 28
            PTP_C_STEP 1 means c_step out 1

            PTP_IACT_CTRL 0 means Input activation from external(ddr), no sparse format, no input convert no output convert
            PTP_WEIGHT_SPARSE 0 no sparse
            PTP_ACT_C_CTRL 0x00400040 input channel 64 output channel 64
            PTP_ACT_W_CTRL 0x00380071 input width(with pad) 113, output width 56
            PTP_ACT_H_CTRL 0x00380071 input height(with pad) 113, output height 56

            // not use unb these value is zero
            PTP_PAD_UnB_ADDR 0
            PTP_WEIGHT_UnB_ADDR 0
            PTP_BIAS_UnB_ADDR 0
            PTP_SCALE_UnB_ADDR 0
            PTP_FP_PARAM_UnB_ADDR 0
            PTP_ROIP_UnB_ADDR 0


            PTP_IACT_W_STRIDE 0x00000e00 width_stride 3584, (C32 format 32 * 112)
            PTP_IACT_SURF_STRIDE 0x00062000 surface_stride 401408 (32 * 112 * 112)
            PTP_IACT_TOTAL_STRIDE 0x000c4000 64 * 112 * 112
            PTP_IACT_BATCH_STRIDE 0x000c4000 in this case it equals to total_stride 64 * 112 * 112

            // not use compress set zero
            PTP_IACT_CTAG_SURF_STRIDE 0
            PTP_IACT_CTAG_BATCH_STRIDE 0

            """
            val = S.uint32x8(0)
            val[0] = 0x001E0620
            val[1] = 0x00000012
            val[2] = 0x0000221B
            val[3] = 0x00010001
            ctrl_desc[4] = val

            val = S.uint32x8(0)
            val[2] = 0x00710070
            val[3] = 0x00390038
            val[4] = 0x001C0038
            val[5] = 0x00000001
            ctrl_desc[5] = val

            val = S.uint32x8(0)
            val[0] = 0x00400040
            val[1] = 0x00380071
            val[2] = 0x00380071
            ctrl_desc[6] = val

            val = S.uint32x8(0)
            val[1] = 0x00000E00
            val[2] = 0x00062000
            val[3] = 0x000C4000
            val[4] = 0x000C4000
            ctrl_desc[7] = val

            """
            9 register, start from 0x40
            Z5_WRB_MODE_CTRL 0x02802102 means region 1 enable, External data format is different and is NHWC
                    Working region 1 acts as output buffer and Working region 0 does not act as output buffer.
                    region0 and region1 not act as internal bufffer
                    use 64k wrb region1 size, region1 alloc to PTP
            WRB_Region0_OACT_CTRL 0 No output activation sparse compression.
            WRB_Region0_OACT_L_STRIDE 0
            WRB_Region0_OACT_S_STRIDE 0
            WRB_Region0_INT_LS_STRIDE 0
            WRB_Region0_OCTAG_S_STRIDE 0
            WRB_Region0_OACT_BATCH_STRIDE 0
            WRB_Region0_OCTAG_BATCH_STRIDE 0
            WRB_Region0_OACT_WH_OFFS 0
            """

            val = S.uint32x8(0)
            val[0] = 0x00090040
            val[1] = 0x02802102
            ctrl_desc[8] = val

            val = S.uint32x8(0)
            ctrl_desc[9] = val

            """
            9 register, start from 0x50
            Z5_WRB_REGION1_PLACEHOLDER 0
            WRB_Region1_OACT_CTRL 0
            WRB_Region1_OACT_L_STRIDE 0x00000040 output line_stride 64
            WRB_Region1_OACT_S_STRIDE 0x00000e00 output surface_stride 56 * 64
            WRB_Region1_INT_LS_STRIDE 0x00380620 int_surface_stride 1568 (28 * 56 w_step_out * h_step_out)
                                                int_line_stride 56

            WRB_Region1_OCTAG_S_STRIDE 0
            WRB_Region1_OACT_BATCH_STRIDE 0
            WRB_Region1_OCTAG_BATCH_STRIDE 0
            WRB_Region1_OACT_WH_OFFS 0
            """
            val = S.uint32x8(0)
            val[0] = 0x00090050
            val[3] = 0x00000040
            val[4] = 0x00000E00
            val[5] = 0x00380620
            ctrl_desc[10] = val

            val = S.uint32x8(0)
            ctrl_desc[11] = val

            """
            4 register from 0x20
            UNB MTP0/MTP1/ITP/PTP base address set to 0
            """
            val = S.uint32x8(0)
            val[0] = 0x00040020
            ctrl_desc[12] = val

            out[0] = 0
            S.aiff(ctrl_desc, param_desc, act_desc)

    return test_aiff_max_pooling


@pytest.mark.X2_1204
def test_aiff():
    # only support uint8
    dtype = "uint8"
    inp = rand((1, 2, 112, 112, 32), dtype)
    gt_out = get_maxpool_gt(inp)

    f_aiff = gen_aiff_maxpool()
    bm = aipu.tir.BuildManager()
    ex = bm.build(f_aiff)

    aipu_out = np.empty((1, 56, 56, 64), dtype=dtype)
    ex(inp, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_aiff()
