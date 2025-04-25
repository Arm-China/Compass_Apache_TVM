# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing


@S.prim_func
def get_bool_from_mask(inp: S.ptr("int32x8", "global")) -> S.bool:
    mask = inp[0] == 0
    return mask[0]


@S.prim_func
def mask_set_get_element_func(inp: S.ptr("int32x8", "global"), out: S.ptr("int32", "global")):
    mask = inp[0] == 1
    out[0:5] = 99

    if get_bool_from_mask(inp):
        out[0] = 10

    mask[1] = False
    if not mask[1]:
        out[1] = 20

    if mask[2] or mask[4]:
        out[2] = 30

    mask2 = mask[3]
    if mask2:
        out[3] = 40

    mask3 = mask[3] and mask[4]
    if mask3:
        out[4] = 50

    cur_out = out + 5
    cur_out[0:8] = S.vsel(S.i32x8(1), 0, mask)


def test_mask_set_get_element():
    dtype = "int32"
    mask = np.array([True, False] * 4)
    a = mask.astype(dtype)

    gt_01234 = [99, 20, 30 if mask[2] or mask[4] else 99, 40 if mask[3] else 99, 50 if mask[3] and mask[4] else 99]
    gt_out = np.concatenate([np.array(gt_01234), np.where(mask, 1, 0)]).astype(dtype)

    bm = aipu.tir.BuildManager()
    ex = bm.build(mask_set_get_element_func)

    py_out = np.empty(13, dtype=dtype)
    mask_set_get_element_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(13, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_mask_set_get_element()
