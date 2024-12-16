# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu, get_range
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_horizontal_op_func(op_name, dtype, hw_lanes):
    sdot_func = getattr(S, op_name)
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def horizontal_op_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(sdot_func(va0, vb0), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(sdot_func(va1, vb1), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(sdot_func(va2, vb2), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(sdot_func(va3, vb3), cur_out)

    return horizontal_op_func


def get_gt_out(op_name, a, b, dtype, hw_lanes):
    if op_name == "vaddh":
        op = lambda x, y: x + y
    elif op_name == "vsubh":
        op = lambda x, y: x - y
    elif op_name == "vmaxh":
        op = max
    else:
        assert op_name == "vminh"
        op = min

    def _compute(x, y):
        x = x.astype("int64")
        y = y.astype("int64")
        return np.clip(op(x, y), *get_range(dtype)).astype(dtype)

    def _execute_once(cur_a, cur_b):
        cur_n = len(cur_a)
        cur_c = np.zeros(cur_n, dtype=dtype)
        for i in range(cur_n // 2):
            cur_c[i] = _compute(cur_a[2 * i], cur_a[2 * i + 1])
            cur_c[i + cur_n // 2] = _compute(cur_b[2 * i], cur_b[2 * i + 1])
        return cur_c

    n = len(a)
    c = np.zeros(n, dtype=dtype)
    cur_a = a[:hw_lanes]
    cur_b = b[:hw_lanes]
    c[:hw_lanes] = _execute_once(cur_a, cur_b)
    cur_a = a[hw_lanes : hw_lanes * 3]
    cur_b = b[hw_lanes : hw_lanes * 3]
    c[hw_lanes : hw_lanes * 3] = _execute_once(cur_a, cur_b)
    cur_a = a[hw_lanes * 3 : hw_lanes * 6]
    cur_b = b[hw_lanes * 3 : hw_lanes * 6]
    c[hw_lanes * 3 : hw_lanes * 6] = _execute_once(cur_a, cur_b)
    cur_a = a[hw_lanes * 6 :]
    cur_b = b[hw_lanes * 6 :]
    c[hw_lanes * 6 :] = _execute_once(cur_a, cur_b)
    return c


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
@pytest.mark.parametrize("op_name", ("vaddh", "vsubh", "vmaxh", "vminh"))
def test_horizontal_operator(op_name, dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = get_gt_out(op_name, a, b, dtype, hw_lanes)

    py_func = gen_horizontal_op_func(op_name, dtype, hw_lanes)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.zeros(n, dtype=dtype)
    py_func(a, b, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.zeros(n, dtype=dtype)
    ex(a, b, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_horizontal_operator("vaddh", "int32")
    test_horizontal_operator("vsubh", "uint16")
