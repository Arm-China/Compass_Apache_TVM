# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import numpy as np
import pytest
from tvm.compass.dsl import BuildManager, script as S, Aiff
from tvm.compass.dsl.testing import rand, assert_allclose


def get_gt(inp):
    ret = np.empty((1, 75, 75, 32), dtype=np.uint8)
    for i in range(75):
        for j in range(75):
            for c in range(32):
                val = np.max(inp[0, 2 * i : 2 * i + 1, 2 * j : 2 * j + 1, c])
                ret[0, i, j, c] = val + 1
    return ret


@S.prim_func
def aiff_max_pool_add_func(
    out: S.ptr("u8", "global"),
    ctrl_desc: S.ptr("u32", "global"),
    param_desc: S.ptr("u32", "global"),
    act_desc: S.ptr("u32", "global"),
):
    if S.get_local_id() != 0:
        return

    S.aiff(ctrl_desc, param_desc, act_desc)

    for i in range(75 * 75 * 32):
        out[i] += 1


def get_aiff(inp):
    # fmt: off
    ctrl = np.array([0x0] * 104, dtype="uint32")
    ctrl[[0, 3, 8, 16, 17, 24, 32, 33, 34, 42, 43, 44, 45, 48, 49, 50]] = (0x81, 0xD, 0x20000, 0x10170, 0x3, 0x70220, 0x1E0620, 0x2, 0x2209, 0x950095, 0x950095, 0x4B004B, 0x1, 0x200020, 0x4B0095, 0x4B0095)
    ctrl[[57, 58, 60, 64, 65, 80, 83, 84, 87, 96]] = (0x12C0, 0xAFC80, 0xAFC80, 0x90040, 0x22100002, 0x90050, 0x960, 0x2BF20, 0x2BF20, 0x40020)
    # fmt: on

    aiff = Aiff(descriptor=ctrl)
    aiff.ptp.iact_addr = inp
    return aiff


def get_desc(aiff, out):
    aiff.wrb.region1_oact_addr = out

    desc_chain_arr = aiff.gen_descriptor()
    return desc_chain_arr.ctrl, desc_chain_arr.param, desc_chain_arr.act


@pytest.mark.X2
def test_aiff():
    dtype = "uint8"
    inp = rand((1, 150, 150, 32), dtype)
    gt_out = get_gt(inp)
    aiff = get_aiff(inp)

    bm = BuildManager()
    ex = bm.build(aiff_max_pool_add_func)

    py_out = np.empty((1, 75, 75, 32), dtype=dtype)
    ctrl, param, act = get_desc(aiff, py_out)
    aiff_max_pool_add_func(py_out, ctrl, param, act)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty((1, 75, 75, 32), dtype=dtype)
    ctrl, param, act = get_desc(aiff, npu_out)
    ex(npu_out, ctrl, param, act)
    assert_allclose(npu_out, gt_out)


@pytest.mark.NOT_X1
@pytest.mark.NOT_X2
def test_aiff_get_none_type_register():
    with pytest.raises(AssertionError) as exc_info:

        # fmt: off
        ctrl = np.array([0x0] * 104, dtype="uint32")
        ctrl[[0, 3, 8, 16, 17, 24, 32, 33, 34, 42, 43, 44, 45, 48, 49, 50]] = (0x81, 0xD, 0x20000, 0x10170, 0x3, 0x70220, 0x1E0620, 0x2, 0x2209, 0x950095, 0x950095, 0x4B004B, 0x1, 0x200020, 0x4B0095, 0x4B0095)
        ctrl[[57, 58, 60, 64, 65, 80, 83, 84, 87, 96]] = (0x12C0, 0xAFC80, 0xAFC80, 0x90040, 0x22100002, 0x90050, 0x960, 0x2BF20, 0x2BF20, 0x40020)
        # fmt: on

        Aiff(target="X3P_1304", descriptor=ctrl)

    exc_msg = str(exc_info.value)
    expect = "The register with address '0' could not found in the AIFF function units for target"
    assert expect in exc_msg, f"\nExpect snippet:\n{expect}\n\nException Message:\n{exc_msg}\n"


if __name__ == "__main__":
    test_aiff()
    test_aiff_get_none_type_register()
