# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_vconcat_func(part, vdtype):
    @S.prim_func
    def vconcat_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        c[0] = S.vconcat(a[0], b[0], part)

    return vconcat_func


def get_vconcat_gt(a, b, part):
    half = len(a) // 2
    if part == "low":
        return np.concatenate((a[:half], b[:half]))
    elif part == "high":
        return np.concatenate((a[half:], b[half:]))
    elif part == "even":
        return np.concatenate((a[::2], b[::2]))
    elif part == "odd":
        return np.concatenate((a[1::2], b[1::2]))
    else:
        assert False, f'Unsupported part "{part}"'


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("part", ("low", "high", "even", "odd"))
def test_vconcat(dtype, part):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = get_vconcat_gt(a, b, part)

    prim_func = gen_vconcat_func(part, vdtype)
    bm = aipu.tir.BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_vconcat("int8", part="low")
    test_vconcat("int8", part="high")
    test_vconcat("int8", part="even")
    test_vconcat("int8", part="odd")
