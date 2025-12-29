# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import get_range
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def get_scalar_gt_out(op_name, a, b, lanes, dtype, mask):
    out = np.zeros(lanes * 2, dtype=dtype)

    if op_name == "add":
        out = np.add(a, b)
    elif op_name == "sub":
        out = np.subtract(a, b)
    elif op_name == "mul":
        out = np.multiply(a, b)
    else:
        assert op_name == "div"
        if dtype.startswith("float"):
            out = a / b
        else:
            out = np.where(b != 0, np.clip(a / b, *get_range(dtype)), 0)

    return np.where(mask, out, 0).astype(dtype)


def gen_scalar_func(op_name, n, lanes, dtype, mask):
    @S.prim_func
    def scalar_add_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        c[0:lanes] = a[0:lanes] + b[0:lanes]
        c[lanes:n] = S.vadd(a[lanes:n], b[lanes:n], mask=mask)

    @S.prim_func
    def scalar_sub_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        c[0:lanes] = a[0:lanes] - b[0:lanes]
        c[lanes:n] = S.vsub(a[lanes:n], b[lanes:n], mask=mask)

    @S.prim_func
    def scalar_mul_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        c[0:lanes] = a[0:lanes] * b[0:lanes]
        c[lanes:n] = S.vmul(a[lanes:n], b[lanes:n], mask=mask)

    @S.prim_func
    def scalar_div_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        c[0:lanes] = a[0:lanes] / b[0:lanes]
        c[lanes:n] = S.vdiv(a[lanes:n], b[lanes:n], mask=mask)

    return locals()[f"scalar_{op_name}_func"]


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("op_name", ("add", "sub", "mul", "div"))
def test_scalar_flexible_width_vector(op_name, dtype):
    if op_name == "mul" and dtype.endswith("8"):
        pytest.skip("The 8bit equal-width multiply is meaningless.")
    if op_name == "div" and dtype == "float16":
        pytest.skip("'S.vdiv' only support integer and float32 instruction.")

    lanes = hw_native_vdtype(dtype).lanes + 3
    n = lanes * 2
    a = rand(n, dtype)
    b = rand(n, dtype)
    if op_name == "div":
        b = np.where(b != 0, b, 1)
    mask = np.concatenate((np.array([True] * lanes), rand(lanes, "bool")))
    gt_out = get_scalar_gt_out(op_name, a, b, lanes, dtype, mask)

    py_func = gen_scalar_func(op_name, n, lanes, dtype, mask[lanes:])
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    assert_allclose(py_out[mask], gt_out[mask])

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


def gen_fma_vector(n, dtype, mask):
    @S.prim_func
    def vfma_func(
        acc: S.ptr(dtype, "global"),
        a: S.ptr(dtype, "global"),
        b: S.ptr(dtype, "global"),
        c: S.ptr(dtype, "global"),
    ):
        c[0:n] = S.fma(acc[0:n], a[0:n], b[0:n], mask)

    return vfma_func


def get_vfma_gt_out(a, b, acc, mask):
    for i in range(len(acc)):
        if mask[i]:
            acc[i] += a[i] * b[i]
    return acc


def test_fma_vector():
    dtype = "float32"
    n = hw_native_vdtype(dtype).lanes + 3
    a = rand(n, dtype)
    b = rand(n, dtype)
    mask = rand(n, "bool")
    acc = np.ones(n, dtype=dtype)
    gt_out = get_vfma_gt_out(a, b, acc, mask)

    py_func = gen_fma_vector(n, dtype, mask)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    acc_py = np.ones(n, dtype=dtype)
    py_func(acc_py, a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=dtype)
    acc_npu = np.ones(n, dtype=dtype)
    ex(acc_npu, a, b, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_vmod_func(dtype, hw_lanes):
    lanes0 = hw_lanes // 2
    lanes1 = hw_lanes - 5
    lanes2 = hw_lanes + 3
    lanes3 = 2 * hw_lanes + 1
    lanes4 = 4 * hw_lanes

    @S.prim_func
    def vmod_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n / 2
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(S.vmod(va0, vb0), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0
        # 2. n - 5
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(S.vmod(va1, vb1), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1
        # 3. n + 3
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(S.vmod(va2, vb2), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2
        # 4. 2 * n + 1
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(S.vmod(va3, vb3), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes3, cur_b + lanes3, cur_out + lanes3
        # 5. 4 * n
        va4 = S.vload(cur_a, lanes=lanes4)
        vb4 = S.vload(cur_b, lanes=lanes4)
        S.vstore(S.vmod(va4, vb4), cur_out)

    return vmod_func


def get_vmod_gt_out(a, b, mask, dtype):
    a = np.array(a, dtype="int64")
    b = np.array(b, dtype="int64")
    n = len(a)
    ret = []
    for i in range(n):
        if b[i] == 0:
            ret.append(0)
        else:
            x = abs(a[i]) % abs(b[i])
            if a[i] < 0:
                ret.append(-x)
            else:
                ret.append(x)
    return np.where(mask, ret, 0).astype(dtype)


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32"))
def test_vmod(dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = int(8.5 * hw_lanes - 1)
    a = rand(n, dtype)
    b = rand(n, dtype, low=0, high=vdtype.bits)
    mask = rand(n, "bool")
    gt_out = get_vmod_gt_out(a, b, mask, dtype)

    py_func = gen_vmod_func(dtype, hw_lanes)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    assert_allclose(py_out[mask], gt_out[mask])

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out[mask], gt_out[mask])


def gen_clip_func(dtype, hw_lanes, min_val, max_val):
    lanes0 = hw_lanes // 2
    lanes1 = hw_lanes - 5
    lanes2 = hw_lanes + 3
    lanes3 = 2 * hw_lanes + 1
    lanes4 = 4 * hw_lanes

    @S.prim_func
    def clip_func(a: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        cur_a, cur_out = a, out
        # 1. n / 2
        va0 = S.vload(cur_a, lanes=lanes0)
        S.vstore(S.clip(va0, min_val, max_val), cur_out)

        cur_a, cur_out = cur_a + lanes0, cur_out + lanes0
        # 2. n - 5
        va1 = S.vload(cur_a, lanes=lanes1)
        S.vstore(S.clip(va1, min_val, max_val), cur_out)

        cur_a, cur_out = cur_a + lanes1, cur_out + lanes1
        # 3. n + 3
        va2 = S.vload(cur_a, lanes=lanes2)
        S.vstore(S.clip(va2, min_val, max_val), cur_out)

        cur_a, cur_out = cur_a + lanes2, cur_out + lanes2
        # 4. 2 * n + 1
        va3 = S.vload(cur_a, lanes=lanes3)
        S.vstore(S.clip(va3, min_val, max_val), cur_out)

        cur_a, cur_out = cur_a + lanes3, cur_out + lanes3
        # 5. 4 * n
        va4 = S.vload(cur_a, lanes=lanes4)
        S.vstore(S.clip(va4, min_val, max_val), cur_out)

    return clip_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_clip(dtype):
    vdtype = hw_native_vdtype(dtype)

    hw_lanes = vdtype.lanes
    n = int(8.5 * hw_lanes - 1)
    a = rand(n, dtype)
    min_val = np.sort(a)[2].tolist()
    max_val = np.sort(a)[n - 2].tolist()
    gt_out = np.clip(a, min_val, max_val)

    py_func = gen_clip_func(dtype, hw_lanes, min_val, max_val)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_scalar_flexible_width_vector("add", "int32")
    test_scalar_flexible_width_vector("sub", "int32")
    test_scalar_flexible_width_vector("mul", "int32")
    test_scalar_flexible_width_vector("div", "int32")
    test_fma_vector()
    test_vmod("int32")
    test_clip("float32")
