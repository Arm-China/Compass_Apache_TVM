# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def get_scalar_gt_out(test_id, inp, lanes, dtype, stride, mask):
    vdtype = hw_native_vdtype(dtype)
    n = inp.size
    out = np.array([99] * n, dtype=dtype)

    if test_id == 0:
        out[:25] = inp[0]
        out[10:17] = 1
        out[25:50:stride] = inp[1]
        out[60:n:stride] = inp[9 + 4]
    elif test_id == 1:
        start_idx = int(np.clip(inp[0], 1, 10))
        out[start_idx : start_idx + 10] = inp[3]
        out[start_idx + 10 : start_idx + 30 : stride] = inp[3]
    elif test_id == 2:
        out[:25] = inp[:25]
        out[25:50:stride] = inp[25:50:stride]
        out[50 + vdtype.lanes : 50 + 4 * vdtype.lanes] = inp[: 3 * vdtype.lanes]
    elif test_id == 3:
        out[:lanes] = np.where(mask, inp[:lanes], out[:lanes])
        vb = inp[lanes : lanes * 3 : 2]
        out[lanes : lanes * 2] = np.where(vb > 0, vb, out[lanes : lanes * 2])
    else:
        assert test_id == 4
        out[:lanes] = np.where(mask, inp[0:lanes], 0)
        out[lanes : lanes * 2] = np.where(inp[:lanes] > 0, inp[lanes : lanes * 2], 0)
    return out.astype(dtype)


def gen_scalar_func(test_id, n, lanes, dtype, stride, mask):
    vdtype = hw_native_vdtype(dtype)

    @S.prim_func
    def scalar_f0(inp: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        out[0:n] = 99

        out[:25] = inp[0]
        out[10:17] = 1
        out[25:50:stride] = inp[1]
        inp_moved = inp + 9
        out[60:n:stride] = inp_moved[4]

    @S.prim_func
    def scalar_f1(inp: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        out[0:n] = 99

        start_idx = S.i32(S.clip(inp[0], min_val=1, max_val=10))
        out[start_idx : start_idx + 10] = inp[3]
        out[start_idx + 10 : start_idx + 30 : stride] = inp[3]

    @S.prim_func
    def scalar_f2(inp: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        out[0:n] = 99

        out[:25] = inp[:25]
        out[25:50:stride] = inp[25:50:stride]

        vinp = inp.as_ptr(vdtype)
        vout = (out + 50).as_ptr(vdtype)
        var = vinp[:3]
        (vout + 1)[:3] = var

    @S.prim_func
    def scalar_f3(inp: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        out[0:n] = 99

        va = inp[:lanes]
        S.vstore(va, out, mask=mask)

        vb = inp[lanes : lanes * 3 : 2]
        S.vstore(vb, out + lanes, mask=(vb > 0))

    @S.prim_func
    def scalar_f4(inp: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        out[0:n] = 99

        va = S.vload(inp, lanes=lanes, mask=mask)
        S.vstore(va, out)

        vb = S.vload(inp + lanes, lanes=lanes, mask=(inp[:lanes] > 0))
        S.vstore(vb, out + lanes)

    return locals()[f"scalar_f{test_id}"]


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32", "float16"))
@pytest.mark.parametrize("test_id", range(5))
def test_scalar_flexible_width_vector(dtype, test_id):
    lanes = hw_native_vdtype(dtype).lanes + 3
    n = 200
    a = rand(n, dtype)
    stride = 2
    mask = rand(lanes, "bool")

    gt_out = get_scalar_gt_out(test_id, a, lanes, dtype, stride, mask)
    py_func = gen_scalar_func(test_id, n, lanes, dtype, stride, mask)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_scalar_flexible_width_vector("int16", 3)
    test_scalar_flexible_width_vector("int32", 3)
