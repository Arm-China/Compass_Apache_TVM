# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import os
import numpy as np
import pytest
from tvm.compass.dsl import BuildManager, script as S, Aiff
from tvm.compass.dsl.testing import assert_allclose


@S.prim_func
def aiff_activation_elu_func(ctrl: S.ptr("u32", "global"), param: S.ptr("u32", "global"), act: S.ptr("u32", "global")):
    if S.get_local_id() != 0:
        return

    ev0, ev1 = S.alloc_events(2)

    S.async_aiff(ctrl, param, act, ev0)
    S.async_aiff(ctrl + 352, param + 88, act + 72, ev1)

    S.wait_events(ev0, ev1)
    S.free_events(ev0, ev1)


def get_aiff(lut, tgt):
    # fmt: off
    ctrl = np.array([0x0] * 352, dtype="uint32")
    ctrl[[0, 3, 8, 9, 16, 17, 18, 19, 23, 24, 31, 32, 34, 35, 36, 64]] = (0x81, 0x2C, 0x10170, 0x4, 0x2F0120, 0x2, 0x100, 0xD, 0x202020, 0x20202, 0x400, 0x800, 0x200020, 0x200020, 0x20002, 0x2F01A0)
    ctrl[[65, 66, 67, 71, 72, 79, 80, 82, 83, 84, 112, 113, 125, 127, 136, 137]] = (0x42, 0x100, 0xD, 0x202020, 0x20202, 0x400, 0x800, 0x200020, 0x200020, 0x20002, 0x100220, 0x30001, 0x4, 0x4, 0x320320, 0x80000000)
    ctrl[[140, 143, 145, 147, 149, 160, 162, 163, 164, 185, 192, 193, 196, 199, 201, 203]] = (0x80000000, 0x4000, 0x10000, 0x7FFFFFFF, 0x1, 0x1, 0x200002, 0x10020, 0x2200220, 0x1, 0x320420, 0x80000000, 0x80000000, 0x4000, 0x10000, 0x7FFFFFFF)
    ctrl[[205, 216, 218, 219, 220, 241, 248, 249, 296, 297, 299, 300, 304, 307, 308, 312]] = (0x1, 0x1, 0x200002, 0x10020, 0x2200220, 0x1, 0x2A0720, 0xF, 0x60040, 0x30400003, 0x400, 0x800, 0x60050, 0x400, 0x800, 0xA0060)
    ctrl[[328, 344, 346, 347, 348]] = (0xA0070, 0x50020, 0x10000, 0x20000, 0x22440)
    # fmt: on

    aiff = Aiff(descriptor=ctrl, target=tgt)
    aiff.itp.lut_mcfg0 = lut
    aiff.itp.lut_mcfg1 = lut

    return aiff


def get_desc(aiff, inp, out):
    aiff.mtp[0].iact_addr = inp
    aiff.mtp[1].iact_addr = inp[2048:]
    aiff.wrb.region0_oact_addr = out
    aiff.wrb.region1_oact_addr = out[2048:]
    desc_chain_arr0 = aiff.gen_descriptor()

    aiff.mtp[0].iact_addr = inp[4096:]
    aiff.mtp[1].iact_addr = inp[6144:]
    aiff.wrb.region0_oact_addr = out[4096:]
    aiff.wrb.region1_oact_addr = out[6144:]
    desc_chain_arr1 = aiff.gen_descriptor()

    return desc_chain_arr0 + desc_chain_arr1


@pytest.mark.NOT_X1
@pytest.mark.NOT_X2
def test_aiff_activation_elu():
    tgt = "X3P_1304"
    dtype = "uint8"
    out_dtype = "int8"
    shape = (8192,)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(cur_dir, "../../../data/aiff_activation_elu"))
    inp = np.fromfile(f"{data_dir}/input0.bin", dtype=dtype).reshape(shape)
    lut = np.fromfile(f"{data_dir}/weight.bin", dtype=dtype)
    gt_out = np.fromfile(f"{data_dir}/gt.bin", dtype=out_dtype).reshape(shape)
    aiff = get_aiff(lut, tgt)

    bm = BuildManager(target=tgt)
    ex = bm.build(aiff_activation_elu_func)

    py_out = np.empty(shape, dtype=out_dtype)
    desc = get_desc(aiff, inp, py_out)
    aiff_activation_elu_func(desc.ctrl, desc.param, desc.act)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(shape, dtype=out_dtype)
    desc = get_desc(aiff, inp, npu_out)
    ex(desc.ctrl, desc.param, desc.act)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_aiff_activation_elu()
