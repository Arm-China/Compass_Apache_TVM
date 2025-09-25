# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import itertools
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def get_gt_output(x, y, part, is_xy_mask):
    if is_xy_mask:
        x, y = x > 0, y > 0

    count = len(x) // 2
    if part == "low":
        gt_out = itertools.chain.from_iterable(zip(x[:count], y[:count]))
    elif part == "high":
        gt_out = itertools.chain.from_iterable(zip(x[count:], y[count:]))
    elif part == "even":
        gt_out = itertools.chain.from_iterable(zip(x[::2], y[::2]))
    elif part == "odd":
        gt_out = itertools.chain.from_iterable(zip(x[1::2], y[1::2]))
    else:
        assert False, "See test any lenth vector zip case if part is all."

    return np.array(list(gt_out), dtype=x.dtype)


def gen_vzip_gentype(part, vdtype):
    @S.prim_func
    def vzip_gentype_func(x: S.ptr(vdtype, "global"), y: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vzip(x[0], y[0], part)

    return vzip_gentype_func


@pytest.mark.parametrize(
    "dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32", "bfloat16")
)
@pytest.mark.parametrize("part", ("low", "high", "even", "odd"))
def test_vzip_gentype(dtype, part):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    x = rand(n, dtype)
    y = rand(n, dtype)
    gt_out = get_gt_output(x, y, part, is_xy_mask=False)

    f_vzip = gen_vzip_gentype(part, vdtype)
    bm = BuildManager()
    ex = bm.build(f_vzip)

    py_out = np.empty(n, dtype)
    f_vzip(x, y, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(x, y, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_vzip_pgentype(part, vdtype):
    @S.prim_func
    def vzip_pgentype_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        mask_a = a[0] > 0
        mask_b = b[0] > 0
        mask_out = S.vzip(mask_a, mask_b, part)

        out[0] = S.vsel(a[0], b[0], mask_out)

    return vzip_pgentype_func


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32"))
@pytest.mark.parametrize("part", ("low", "high", "even", "odd"))
def test_vzip_pgentype(dtype, part):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_tmp = get_gt_output(a, b, part, is_xy_mask=True)
    gt_out = np.where(gt_tmp, a, b)

    f_vzip = gen_vzip_pgentype(part, vdtype)
    bm = BuildManager()
    ex = bm.build(f_vzip)

    py_out = np.empty(n, dtype)
    f_vzip(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vzip_gentype(dtype="float32", part="low")
    test_vzip_pgentype(dtype="int8", part="high")
    test_vzip_gentype(dtype="float16", part="even")
    test_vzip_pgentype(dtype="int16", part="odd")
