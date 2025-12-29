# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import get_range
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


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
        cur_c = np.empty(cur_n, dtype=dtype)
        for i in range(cur_n // 2):
            cur_c[i] = _compute(cur_a[2 * i], cur_a[2 * i + 1])
            cur_c[i + cur_n // 2] = _compute(cur_b[2 * i], cur_b[2 * i + 1])
        return cur_c

    n = len(a)
    c = np.empty(n, dtype=dtype)
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
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dtype)
    py_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_vrpadd_func(in_dtype, out_dtype, hw_lanes):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def vrpadd_func(a: S.ptr(in_dtype, "global"), out: S.ptr(out_dtype, "global")):
        cur_a, cur_out = a, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        S.vstore(S.vrpadd(va0), cur_out)

        cur_a, cur_out = cur_a + lanes0, cur_out + lanes0
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        S.vstore(S.vrpadd(va1), cur_out)

        cur_a, cur_out = cur_a + lanes1, cur_out + lanes1
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        S.vstore(S.vrpadd(va2), cur_out)

        cur_a, cur_out = cur_a + lanes2, cur_out + lanes2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        S.vstore(S.vrpadd(va3), cur_out)

    return vrpadd_func


def get_vrpadd_gt_out(a, hw_lanes, out_dtype):
    a = a.astype(out_dtype)
    ret = np.zeros_like(a)
    ret[0] = np.sum(a[:hw_lanes])
    ret[hw_lanes] = np.sum(a[hw_lanes : 3 * hw_lanes])
    ret[3 * hw_lanes] = np.sum(a[3 * hw_lanes : 6 * hw_lanes])
    ret[6 * hw_lanes] = np.sum(a[6 * hw_lanes :])
    return ret


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vrpadd(dtype):
    vdtype = hw_native_vdtype(dtype)
    hw_lanes = vdtype.lanes
    out_dtype = "float32" if dtype == "float16" else dtype

    n = 10 * hw_lanes
    a = rand(n, dtype, low=0, high=127)
    gt_out = get_vrpadd_gt_out(a, hw_lanes, out_dtype)

    py_func = gen_vrpadd_func(dtype, out_dtype, hw_lanes)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=out_dtype)
    py_func(a, py_out)
    assert_allclose(py_out[0], gt_out[0])
    assert_allclose(py_out[hw_lanes], gt_out[hw_lanes])
    assert_allclose(py_out[3 * hw_lanes], gt_out[3 * hw_lanes])
    assert_allclose(py_out[6 * hw_lanes], gt_out[6 * hw_lanes])

    npu_out = np.empty(n, dtype=out_dtype)
    ex(a, npu_out)
    assert_allclose(npu_out[0], gt_out[0])
    assert_allclose(npu_out[hw_lanes], gt_out[hw_lanes])
    assert_allclose(npu_out[3 * hw_lanes], gt_out[3 * hw_lanes])
    assert_allclose(npu_out[6 * hw_lanes], gt_out[6 * hw_lanes])


@pytest.mark.parametrize("dtype", ("int32", "uint32", "float32"))
def gen_vrpadd_add_func(in_dtype, out_dtype, hw_lanes):
    @S.prim_func
    def vrpadd_add_func(a: S.ptr(in_dtype, "global"), b: S.ptr(in_dtype, "global"), out: S.ptr(out_dtype, "global")):
        va = S.vload(a, lanes=2 * hw_lanes)
        vb = S.vload(b, lanes=2 * hw_lanes)
        sum_va = S.vrpadd(va)
        vc = S.vadd(sum_va, vb)  # sum_va ret_dtype should be same as vb's dtype : with lanes = 2 * hw_lanes
        S.vstore(vc, out)

    return vrpadd_add_func


def get_vrpadd_add_gt_out(a, b, n, out_dtype):
    a = a.astype(out_dtype)
    ret = np.zeros_like(a)
    ret[0] = np.sum(a)
    return ret[0] + b[0]


def test_vrpadd_rettype(dtype="int32"):
    vdtype = hw_native_vdtype(dtype)
    hw_lanes = vdtype.lanes
    out_dtype = "float32" if dtype == "float16" else dtype

    n = 2 * hw_lanes
    a = rand(n, dtype, low=0, high=127)
    b = rand(n, dtype, low=0, high=127)
    gt_out = get_vrpadd_add_gt_out(a, b, n, out_dtype)

    py_func = gen_vrpadd_add_func(dtype, out_dtype, hw_lanes)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=out_dtype)
    py_func(a, b, py_out)
    assert_allclose(py_out[0], gt_out)

    npu_out = np.empty(n, dtype=out_dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out[0], gt_out)


def gen_vdpa_func(in_dtype, in1_dtype, out_dtype, hw_lanes):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def vdpa_func(a: S.ptr(in_dtype, "global"), b: S.ptr(in1_dtype, "global"), out: S.ptr(out_dtype, "global")):
        if S.get_local_id() != 0:
            return

        cur_a, cur_b, cur_out = a, b, out
        # 1. n
        acc0 = S.cast(1, f"{out_dtype}x{lanes0 // 2}")
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(S.vdpa(acc0, va0, vb0), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0 // 2
        # 2. 2 * n
        acc1 = S.cast(1, f"{out_dtype}x{lanes1 // 2}")
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(S.vdpa(acc1, va1, vb1), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1 // 2
        # 3. 3 * n
        acc2 = S.cast(1, f"{out_dtype}x{lanes2 // 2}")
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(S.vdpa(acc2, va2, vb2), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2 // 2
        # 4. 4 * n
        acc3 = S.cast(1, f"{out_dtype}x{lanes3 // 2}")
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(S.vdpa(acc3, va3, vb3), cur_out)

    return vdpa_func


def get_vdpa_gt(a, b, acc):
    out = [0] * len(acc)
    a = a.astype("int64")
    b = b.astype("int64")
    for i in range(len(acc)):
        out[i] = acc[i] + a[2 * i] * b[2 * i] + a[2 * i + 1] * b[2 * i + 1]
    return np.clip(out, *get_range(acc.dtype)).astype(acc.dtype)


@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    (
        ("int8", "int8", "int16"),
        ("int8", "uint8", "int16"),
        ("uint8", "int8", "int16"),
        ("uint8", "uint8", "uint16"),
        ("int16", "int16", "int32"),
        ("int16", "uint16", "int32"),
        ("uint16", "int16", "int32"),
        ("uint16", "uint16", "uint32"),
    ),
)
def test_vdpa(in0_dtype, in1_dtype, out_dtype):
    vdtype = hw_native_vdtype(in0_dtype)
    hw_lanes = vdtype.lanes
    n = hw_lanes * 10
    a = rand(n, dtype=in0_dtype)
    b = rand(n, dtype=in1_dtype)
    acc = np.ones(n // 2, dtype=out_dtype)
    gt_out = acc.copy()
    gt_out = get_vdpa_gt(a, b, gt_out)

    py_func = gen_vdpa_func(in0_dtype, in1_dtype, out_dtype, hw_lanes)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = acc.copy()
    py_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = acc.copy()
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_vqdpa_func(in_dtype, in1_dtype, out_dtype, hw_lanes, n_multiple):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def vqdpa_func(a: S.ptr(in_dtype, "global"), b: S.ptr(in1_dtype, "global"), out: S.ptr(out_dtype, "global")):
        if S.get_local_id() != 0:
            return

        cur_a, cur_b, cur_out = a, b, out
        # 1. n
        acc0 = S.cast(1, f"{out_dtype}x{lanes0 // n_multiple}")
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(S.vqdpa(acc0, va0, vb0), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0 // n_multiple
        # 2. 2 * n
        acc1 = S.cast(1, f"{out_dtype}x{lanes1 // n_multiple}")
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(S.vqdpa(acc1, va1, vb1), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1 // n_multiple
        # 3. 3 * n
        acc2 = S.cast(1, f"{out_dtype}x{lanes2 // n_multiple}")
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(S.vqdpa(acc2, va2, vb2), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2 // n_multiple
        # 4. 4 * n
        acc3 = S.cast(1, f"{out_dtype}x{lanes3 // n_multiple}")
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(S.vqdpa(acc3, va3, vb3), cur_out)

    return vqdpa_func


def get_vqdpa_gt(a, b, acc):
    out = acc.copy()
    if len(a) / len(acc) == 4:
        # vqdpa integer
        a = a.astype("int64")
        b = b.astype("int64")
        out = out.astype("int64")
        for i in range(len(acc)):
            for j in range(4):
                xy_idx = 4 * i + j
                out[i] += a[xy_idx] * b[xy_idx]
        out = np.clip(out, *get_range(acc.dtype))
    else:
        # vqdpa float
        a = a.astype("float64")
        b = b.astype("float64")
        out = out.astype("float64")
        for i in range(len(a)):
            out[(i // 4) * 2] += a[i] * b[i]
    return out.astype(acc.dtype)


def _test_vqdpa(in0_dtype, in1_dtype, out_dtype):
    vdtype = hw_native_vdtype(in0_dtype)
    hw_lanes = vdtype.lanes
    n = hw_lanes * 10
    n_multiple = 4 if vdtype.is_integer else 2
    a = rand(n, dtype=in0_dtype)
    b = rand(n, dtype=in1_dtype)
    acc = np.ones(n // n_multiple, dtype=out_dtype)
    gt_out = acc.copy()
    gt_out = get_vqdpa_gt(a, b, gt_out)

    py_func = gen_vqdpa_func(in0_dtype, in1_dtype, out_dtype, hw_lanes, n_multiple)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = acc.copy()
    py_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = acc.copy()
    ex(a, b, npu_out)
    atol = 1 if out_dtype == "float32" else None
    assert_allclose(npu_out, gt_out, atol=atol)


@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    (
        ("int8", "int8", "int32"),
        ("int8", "uint8", "int32"),
        ("uint8", "int8", "int32"),
        ("uint8", "uint8", "uint32"),
    ),
)
def test_integer_vqdpa(in0_dtype, in1_dtype, out_dtype):
    _test_vqdpa(in0_dtype, in1_dtype, out_dtype)


@pytest.mark.NOT_X3P
@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    (
        ("float16", "float16", "float32"),
        ("bfloat16", "bfloat16", "float32"),
    ),
)
def test_floating_vqdpa(in0_dtype, in1_dtype, out_dtype):
    _test_vqdpa(in0_dtype, in1_dtype, out_dtype)


def gen_vdot_func(in0_dtype, in1_dtype, out_dtype, hw_lanes):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def vdot_func(a: S.ptr(in0_dtype, "global"), b: S.ptr(in1_dtype, "global"), out: S.ptr(out_dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(S.vdot(va0, vb0), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0 // 2
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(S.vdot(va1, vb1), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1 // 2
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(S.vdot(va2, vb2), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2 // 2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(S.vdot(va3, vb3), cur_out)

    return vdot_func


def get_vdot_gt_out(a, b, out_n, out_dtype):
    a = a.astype("int64")
    b = b.astype("int64")
    out = [0] * out_n
    for i in range(len(a)):
        out[i // 2] += a[i] * b[i]
    return np.clip(out, *get_range(out_dtype)).astype(out_dtype)


@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    (
        ("int8", "int8", "int16"),
        ("int8", "uint8", "int16"),
        ("uint8", "int8", "int16"),
        ("uint8", "uint8", "uint16"),
        ("int16", "int16", "int32"),
        ("int16", "uint16", "int32"),
        ("uint16", "int16", "int32"),
        ("uint16", "uint16", "uint32"),
    ),
)
def test_vdot(in0_dtype, in1_dtype, out_dtype):
    vdtype = hw_native_vdtype(in0_dtype)

    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    out_n = n // 2
    a = rand(n, in0_dtype)
    b = rand(n, in1_dtype)
    gt_out = get_vdot_gt_out(a, b, out_n, out_dtype)

    py_func = gen_vdot_func(in0_dtype, in1_dtype, out_dtype, hw_lanes)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(out_n, dtype=out_dtype)
    py_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(out_n, dtype=out_dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


def gen_vqdot_func(in0_dtype, in1_dtype, out_dtype, hw_lanes, n_multiple):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def vqdot_func(a: S.ptr(in0_dtype, "global"), b: S.ptr(in1_dtype, "global"), out: S.ptr(out_dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        S.vstore(S.vqdot(va0, vb0), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0 // n_multiple
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        S.vstore(S.vqdot(va1, vb1), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1 // n_multiple
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        S.vstore(S.vqdot(va2, vb2), cur_out)

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2 // n_multiple
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        S.vstore(S.vqdot(va3, vb3), cur_out)

    return vqdot_func


def get_vqdot_gt_out(a, b, out_n, out_dtype):
    is_float = out_dtype == "float32"
    a = a.astype("float64" if is_float else "int64")
    b = b.astype("float64" if is_float else "int64")
    out = np.zeros(out_n)
    if is_float:
        # vqdot float
        for i in range(len(a)):
            out[(i // 4) * 2] += a[i] * b[i]
    else:
        # vqdot integer
        for i in range(out_n):
            for j in range(4):
                xy_idx = 4 * i + j
                out[i] += a[xy_idx] * b[xy_idx]
        out = np.clip(out, *get_range(out_dtype))
    return out.astype(out_dtype)


@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    (
        ("int8", "int8", "int32"),
        ("int8", "uint8", "int32"),
        ("uint8", "int8", "int32"),
        ("uint8", "uint8", "uint32"),
        ("float16", "float16", "float32"),
        ("bfloat16", "bfloat16", "float32"),
    ),
)
def test_vqdot(in0_dtype, in1_dtype, out_dtype):
    vdtype = hw_native_vdtype(in0_dtype)
    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    n_multiple = 4 if vdtype.is_integer else 2
    out_n = n // n_multiple
    a = rand(n, in0_dtype)
    b = rand(n, in1_dtype)
    gt_out = get_vqdot_gt_out(a, b, out_n, out_dtype)
    mask_out = [True, False] * (out_n // 2) if out_dtype == "float32" else [True] * out_n

    py_func = gen_vqdot_func(in0_dtype, in1_dtype, out_dtype, hw_lanes, n_multiple)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(out_n, dtype=out_dtype)
    py_func(a, b, py_out)
    assert_allclose(py_out[mask_out], gt_out[mask_out])

    npu_out = np.empty(out_n, dtype=out_dtype)
    ex(a, b, npu_out)

    # For BF16, detailed see CP-23944.
    rtol = 1e-3 if in0_dtype == "bfloat16" else None
    atol = 1 if in0_dtype == "float16" else None
    assert_allclose(npu_out[mask_out], gt_out[mask_out], rtol=rtol, atol=atol)


if __name__ == "__main__":
    test_horizontal_operator("vaddh", "int32")
    test_horizontal_operator("vsubh", "uint16")
    test_vrpadd("float16")
    test_vrpadd_rettype("int32")
    test_vdpa("int8", "int8", "int16")
    test_floating_vqdpa("float16", "float16", "float32")
    test_floating_vqdpa("bfloat16", "bfloat16", "float32")
    test_vdot("int8", "int8", "int16")
    test_vqdot("float16", "float16", "float32")
    test_vqdot("bfloat16", "bfloat16", "float32")
