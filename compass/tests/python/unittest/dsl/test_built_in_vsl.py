# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import get_range
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vsl_func(vdtype, shift_vdtype, mask, saturate):
    @S.prim_func
    def vsl_func(x: S.ptr(vdtype, "global"), shift: S.ptr(shift_vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vsl(x[0], shift[0], mask[: vdtype.lanes], saturate)
        out[1] = S.vsl(x[1], 1, mask[vdtype.lanes :], saturate)

    return vsl_func


def _vsl(x, shift, mask, saturate, vdtype):
    dtype = vdtype.element_of
    if saturate:
        shift = np.minimum(shift.astype("uint32"), vdtype.bits)
        out = np.clip(x.astype("int64") << shift, *get_range(dtype))
    else:
        out = x << shift
    return np.where(mask, out, 0).astype(dtype)


def get_gt_out(x, shift, mask, saturate, vdtype):
    lanes = vdtype.lanes
    out0 = _vsl(x[:lanes], shift[:lanes], mask[:lanes], saturate, vdtype)
    out1 = _vsl(x[lanes:], np.array(1), mask[lanes:], saturate, vdtype)
    return np.concatenate([out0, out1])


@pytest.mark.parametrize("dtype", ("uint8", "uint16", "uint32", "int8", "int16", "int32"))
@pytest.mark.parametrize("saturate", (True, False))
@pytest.mark.parametrize("is_shift_signed", (True, False))
def test_vsl(dtype, saturate, is_shift_signed):
    vdtype = hw_native_vdtype(dtype)
    if vdtype.is_uint and saturate:
        pytest.skip("Unsupported saturate vsl for unsigned integer")

    shift_vdtype = vdtype.with_int() if is_shift_signed else vdtype.with_uint()
    n = vdtype.lanes * 2

    x = rand(n, dtype)
    min_val, max_val = get_range(dtype)
    x[:5] = [min_val + 1, max_val - 1, 0, max_val, min_val]

    shift = rand(n, shift_vdtype.element_of)
    shift[:7] = [3, -3, 0, 1, -1, vdtype.bits, vdtype.bits + 3]
    np.random.shuffle(shift)
    mask = rand(n, "bool")

    name = f"vsl_{dtype}_{saturate}_{is_shift_signed}"
    gt_out = get_gt_out(x, shift, mask, saturate, vdtype)

    py_func = gen_vsl_func(vdtype, shift_vdtype, mask, saturate)
    bm = BuildManager()
    ex = bm.build(py_func, name=name)

    py_out = np.empty(n, dtype)
    py_func(x, shift, py_out)
    assert_allclose(py_out[mask], gt_out[mask])

    npu_out = np.empty(n, dtype)
    ex(x, shift, npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


if __name__ == "__main__":
    test_vsl("int32", saturate=False, is_shift_signed=True)
    test_vsl("int8", saturate=True, is_shift_signed=True)
