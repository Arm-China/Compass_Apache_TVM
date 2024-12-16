# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu, DataType
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_reinterpret_func(from_dtype, to_dtype, hw_lanes, ratio):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes
    to_lanes0 = int(lanes0 * ratio)
    to_lanes1 = int(lanes1 * ratio)
    to_lanes2 = int(lanes2 * ratio)
    to_lanes3 = int(lanes3 * ratio)

    @S.prim_func
    def reinterpret_func(a: S.ptr(from_dtype, "global"), out: S.ptr(to_dtype, "global")):
        cur_a, cur_out = a, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        S.vstore(S.reinterpret(va0, f"{to_dtype}x{to_lanes0}"), cur_out)

        cur_a, cur_out = cur_a + lanes0, cur_out + to_lanes0
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        S.vstore(S.reinterpret(va1, f"{to_dtype}x{to_lanes1}"), cur_out)

        cur_a, cur_out = cur_a + lanes1, cur_out + to_lanes1
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        S.vstore(S.reinterpret(va2, f"{to_dtype}x{to_lanes2}"), cur_out)

        cur_a, cur_out = cur_a + lanes2, cur_out + to_lanes2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        S.vstore(S.reinterpret(va3, f"{to_dtype}x{to_lanes3}"), cur_out)

    return reinterpret_func


@pytest.mark.parametrize(
    "from_dtype, to_dtype",
    (
        ("uint8", "int8"),
        ("int16", "uint16"),
        ("uint16", "float16"),
        ("float32", "int32"),
        ("uint8", "int32"),
        ("int16", "uint32"),
        ("uint32", "float16"),
        ("float32", "int8"),
    ),
)
def test_reinterpret(from_dtype, to_dtype):
    ratio = DataType(from_dtype).bits / DataType(to_dtype).bits
    vdtype = hw_native_vdtype(from_dtype)

    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    a = rand(n, from_dtype)
    gt_out = a.view(to_dtype)

    py_func = gen_reinterpret_func(from_dtype, to_dtype, hw_lanes, ratio)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(int(n * ratio), dtype=to_dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(int(n * ratio), dtype=to_dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_reinterpret("uint32", "int32")
    test_reinterpret("uint8", "int32")
    test_reinterpret("uint32", "int8")
