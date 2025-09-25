# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import os
import numpy as np
import pytest
from tvm.compass.dsl import BuildManager, script as S, Aiff
from tvm.compass.dsl.testing import assert_allclose


@S.prim_func
def aiff_activation_elu_func(ctrl: S.ptr("u32", "global"), param: S.ptr("u32", "global"), act: S.ptr("u32", "global")):
    if S.get_local_id() != 0:
        return

    S.aiff(ctrl, param, act)


def get_aiff(inp, lut):
    # fmt: off
    ctrl = np.array([0x0] * 248, dtype="uint32")
    ctrl[[0, 3, 8, 9, 10, 16, 17, 24, 25, 27, 31, 32, 33, 41, 42, 44]] = (0x81, 0x1F, 0x20000, 0x100, 0x80000000, 0x10170, 0x4, 0x1C0120, 0x2, 0xD, 0x404040, 0x20202, 0x1000, 0x800, 0x1000, 0x200020)
    ctrl[[45, 46, 56, 57, 59, 63, 64, 65, 73, 74, 76, 77, 78, 88, 89, 96]] = (0x400040, 0x20002, 0x1C01A0, 0x42, 0xD, 0x404040, 0x20202, 0x1000, 0x800, 0x1000, 0x200020, 0x400040, 0x20002, 0x70220, 0x30001, 0x230320)
    ctrl[[97, 100, 103, 104, 106, 108, 119, 121, 122, 123, 136, 137, 140, 143, 144, 146]] = (0x80000000, 0x80000000, 0x2000, 0x100, 0x7FFFFFFF, 0x1, 0x1, 0x400002, 0x10020, 0x2400240, 0x230420, 0x80000000, 0x80000000, 0x2000, 0x100, 0x7FFFFFFF)
    ctrl[[148, 159, 161, 162, 163, 176, 177, 208, 209, 211, 212, 215, 224, 227, 228, 231]] = (0x1, 0x1, 0x400002, 0x10020, 0x2400240, 0x1E0620, 0x7, 0x90040, 0x30400003, 0x800, 0x1000, 0x1000, 0x90050, 0x800, 0x1000, 0x1000)
    ctrl[[240, 242, 243]] = (0x40020, 0x10000, 0x20000)
    # fmt: on

    aiff = Aiff(descriptor=ctrl)
    aiff.itp.lut_mcfg.base_addr = lut

    aiff.mtp[0].iact_addr = inp
    aiff.mtp[1].iact_addr = inp[4096:]

    return aiff


def get_desc(aiff, out):
    aiff.wrb.region0_oact_addr = out
    aiff.wrb.region1_oact_addr = out[4096:]

    return aiff.gen_descriptor()


@pytest.mark.X2
def test_aiff_activation_elu():
    dtype = "uint8"
    out_dtype = "int8"
    shape = (8192,)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(cur_dir, "../../../data/aiff_activation_elu"))
    inp = np.fromfile(f"{data_dir}/input0.bin", dtype=dtype).reshape(shape)
    lut = np.fromfile(f"{data_dir}/weight.bin", dtype=dtype)
    gt_out = np.fromfile(f"{data_dir}/gt.bin", dtype=out_dtype).reshape(shape)
    aiff = get_aiff(inp, lut)

    bm = BuildManager()
    ex = bm.build(aiff_activation_elu_func)

    py_out = np.empty(shape, dtype=out_dtype)
    desc = get_desc(aiff, py_out)
    aiff_activation_elu_func(desc.ctrl, desc.param, desc.act)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(shape, dtype=out_dtype)
    desc = get_desc(aiff, npu_out)
    ex(desc.ctrl, desc.param, desc.act)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_aiff_activation_elu()
