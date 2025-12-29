# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import DataType
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_clz(dtype):
    @S.prim_func
    def clz_func(x: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        out[0] = S.clz(x[0])

    return clz_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_clz_gentype(dtype):
    x = rand((1,), dtype)
    bin_str = np.binary_repr(x[0], DataType(dtype).bits)
    gt_out = np.array([len(bin_str) - len(bin_str.lstrip("0"))], dtype=dtype)

    f_clz = gen_clz(dtype)
    bm = BuildManager()
    ex = bm.build(f_clz)

    py_out = np.empty(1, dtype)
    f_clz(x, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(1, dtype)
    ex(x, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_vclz(vdtype, mask):
    @S.prim_func
    def vclz_func(x: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.clz(x[0], mask)

    return vclz_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_vclz_gentype(dtype):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    x = rand(n, dtype)
    mask = rand(n, "bool")
    x_binary_list = [np.binary_repr(y, vdtype.bits) for y in x.tolist()]
    x_clz_list = np.array([len(s) - len(s.lstrip("0")) for s in x_binary_list], dtype=dtype)
    gt_out = np.where(mask, x_clz_list, 0)

    f_vclz = gen_vclz(vdtype, mask)
    bm = BuildManager()
    ex = bm.build(f_vclz)

    py_out = np.empty(n, dtype)
    f_vclz(x, py_out)
    assert_allclose(py_out[mask], gt_out[mask])

    npu_out = np.empty(n, dtype)
    ex(x, npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


if __name__ == "__main__":
    test_clz_gentype("uint32")
    test_vclz_gentype("int8")
