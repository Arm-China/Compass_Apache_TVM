# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
import pytest
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import rand


def get_gt(inp):
    ret = np.empty((1, 75, 75, 32), dtype=np.uint16)
    for i in range(75):
        for j in range(75):
            for c in range(32):
                val = np.max(inp[0, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2, c])
                ret[0, i, j, c] = val + 1
    ret[0, 0, 0, 10] += 1
    ret[0, 0, 0, 12] += 1
    return ret


@S.prim_func
def aiff_setting_desc_func(
    out: S.ptr("u16", "global"),
    ctrl: S.ptr("u32", "global"),
    param: S.ptr("u32", "global"),
    act: S.ptr("u32", "global"),
):
    if S.get_local_id() != 0:
        return

    ctrl[33] = S.u16(258)
    x = S.reinterpret(out, "uint32")
    x += 1
    x -= 1
    act[42] = x
    S.aiff(ctrl, param, act)

    for i in range(75 * 75 * 32):
        out[i] += 1

    act_out = act[42]
    act_out_ptr = S.reinterpret(act_out, "uint16 *")
    act_out_ptr[10] += 1

    act_out2 = act_out + 2 * 2
    act_out2_ptr = S.reinterpret(act_out2, "uint16 *")
    act_out2_ptr[10] += 1


def get_desc(inp):
    # fmt: off
    ctrl = np.array([0x0] * 104, dtype="uint32")
    ctrl[[0, 3, 8, 16, 17, 24, 32, 34, 42, 43, 44, 45, 48, 49, 50]] = (0x81, 0xD, 0x20000, 0x10170, 0x3, 0x70220, 0x1E0620, 0x2212, 0x960096, 0x320032, 0x19004B, 0x1, 0x200020, 0x4B0096, 0x4B0096)
    ctrl[[57, 58, 60, 64, 65, 80, 83, 84, 85, 87, 96]] = (0x12C0, 0xAFC80, 0x15F900, 0x90040, 0x2802102, 0x90050, 0x40, 0x12C0, 0x4B0753, 0x57E40, 0x40020)
    # fmt: on

    aiff = aipu.tir.Aiff(descriptor=ctrl)
    aiff.ptp.iact_addr = inp
    desc = aiff.gen_descriptor()
    return desc


@pytest.mark.X2_1204
def test_aiff_setting_desc():
    dtype = "uint16"
    inp = rand((1, 150, 150, 32), dtype)
    gt_out = get_gt(inp)
    inp = inp.reshape((1, 150, 150, 2, 16)).transpose((0, 3, 1, 2, 4))
    inpt = inp.copy()
    desc = get_desc(inpt)

    bm = aipu.tir.BuildManager()
    ex = bm.build(aiff_setting_desc_func)

    py_out = np.empty((1, 75, 75, 32), dtype=dtype)
    aiff_setting_desc_func(py_out, desc.ctrl, desc.param, desc.act)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty((1, 75, 75, 32), dtype=dtype)
    ex(aipu_out, desc.ctrl, desc.param, desc.act)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_aiff_setting_desc()
