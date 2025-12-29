# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vfma(instr, inp_vdtype, out_vdtype, mask):
    @S.prim_func
    def vfma_func(
        a: S.ptr(inp_vdtype, "global"),
        b: S.ptr(inp_vdtype, "global"),
        c: S.ptr(out_vdtype, "global"),
    ):
        tid = S.get_local_id()
        acc = S.cast(1, out_vdtype)
        c[tid] = instr(acc, a[tid], b[tid], mask)

    return vfma_func


def gt_fma(a, b, acc, mask, func_name):
    if len(mask) < len(acc):
        mask = np.tile(mask, len(acc) // len(mask))
    out_dtype = acc.dtype
    a = a.astype(out_dtype)
    b = b.astype(out_dtype)

    for i in range(len(acc)):
        if func_name == "vfma":
            xy_idx = i
        elif func_name == "vfmae":
            xy_idx = i * 2
        else:
            xy_idx = i * 2 + 1  # vfmao
        if mask[i]:
            acc[i] += a[xy_idx] * b[xy_idx]
    return acc


@pytest.mark.parametrize(
    "inp_dtype, func_name",
    (
        ("float32", "vfma"),
        ("float16", "vfmae"),
        ("bfloat16", "vfmae"),
        ("float16", "vfmao"),
        ("bfloat16", "vfmao"),
    ),
)
def test_fma_vector(inp_dtype, func_name):
    name2func_table = {"vfma": S.fma, "vfmae": S.vfmae, "vfmao": S.vfmao}
    func = name2func_table[func_name]

    out_dtype = "float32"
    inp_vdtype = hw_native_vdtype(inp_dtype)
    out_vdtype = hw_native_vdtype(out_dtype)
    n, out_n = inp_vdtype.lanes * 4, out_vdtype.lanes * 4
    a = rand(n, inp_dtype)
    b = rand(n, inp_dtype)
    acc = np.ones(out_n, dtype=out_dtype)
    mask = rand(out_vdtype.lanes, "bool")
    gt_out = gt_fma(a, b, acc, mask, func_name)

    prim_func = gen_vfma(func, inp_vdtype, out_vdtype, mask)
    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(out_n, dtype=out_dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(out_n, dtype=out_dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


@pytest.mark.parametrize(
    "inp_dtype, out_dtype, func_name",
    (
        ("float16", "float16", "vfma"),
        ("int16", "int32", "vfmae"),
        ("float32", "float32", "vfmao"),
    ),
)
def test_fma_vector_fail_invalid_dtype(capfd, inp_dtype, out_dtype, func_name):
    instr = {"vfma": S.fma, "vfmae": S.vfmae, "vfmao": S.vfmao}[func_name]

    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func(
            a: S.ptr(hw_native_vdtype(inp_dtype), "global"),
            b: S.ptr(hw_native_vdtype(inp_dtype), "global"),
            c: S.ptr(hw_native_vdtype(out_dtype), "global"),
        ):
            c[0] = instr(c[0], a[0], b[0])

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = "Only support float32 instruction" if func_name == "vfma" else "Argument type mismatch"
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_fma_vector_corner_case():
    dtype = "float32"
    n = hw_native_vdtype(dtype).lanes
    a = np.array([np.uint32(0xC2AF56DB).view("float32")] * n, dtype)
    b = np.array([np.uint32(0x43EC3C00).view("float32")] * n, dtype)
    gt_out = (a.astype("float64") * 5 + b.astype("float64")).astype("float32")

    @S.prim_func
    def prim_func(in0: S.ptr("fp32x8", "global"), in1: S.ptr("fp32x8", "global"), out: S.ptr("fp32x8", "global")):
        out[0] = S.fma(acc=in1[0], x=in0[0], y=5)

    bm = BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype=dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out, rtol=0, atol=0)

    npu_out = np.empty(n, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out, rtol=0, atol=0)


@S.prim_func
def fma_scalar_func(
    acc: S.ptr("fp32", "global"),
    a: S.ptr("fp32", "global"),
    b: S.ptr("fp32", "global"),
    out: S.ptr("fp32", "global"),
):
    for i in range(8):
        out[i] = S.fma(acc[i], a[i], b[i])


def test_fma_scalar():
    dtype = "float32"
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes

    acc = rand(n, dtype)
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = (acc.astype("float64") + a.astype("float64") * b.astype("float64")).astype(dtype)

    bm = BuildManager()
    ex = bm.build(fma_scalar_func)

    py_out = np.empty(n, dtype=dtype)
    fma_scalar_func(acc, a, b, py_out)
    assert_allclose(py_out, gt_out, rtol=0, atol=0)

    npu_out = np.empty(n, dtype=dtype)
    ex(acc, a, b, npu_out)
    assert_allclose(npu_out, gt_out, rtol=0, atol=0)


if __name__ == "__main__":
    test_fma_vector("float32", "vfma")
    test_fma_vector("float16", "vfmae")
    test_fma_vector("float16", "vfmao")
    test_fma_vector_fail_invalid_dtype(None, "int16", "int32", "vfmae")
    test_fma_vector_corner_case()
    test_fma_scalar()
    test_fma_vector("bfloat16", "vfmae")
    test_fma_vector("bfloat16", "vfmao")
