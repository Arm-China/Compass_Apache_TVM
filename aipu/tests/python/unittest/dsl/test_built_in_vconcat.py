# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_vconcat_func(part, dtype, vdtype, inp_num):
    @S.prim_func
    def vconcat2_func(a0: S.ptr(vdtype, "global"), a1: S.ptr(vdtype, "global"), c: S.ptr(vdtype, "global")):
        c[0] = S.vconcat([a0[0], a1[0]], part)

    @S.prim_func
    def vconcat3_func(
        a0: S.ptr(vdtype, "global"),
        a1: S.ptr(vdtype, "global"),
        a2: S.ptr(vdtype, "global"),
        c: S.ptr(dtype, "global"),
    ):
        S.vstore(S.vconcat([a0[0], a1[0], a2[0]], part), c)

    @S.prim_func
    def vconcat4_func(
        a0: S.ptr(vdtype, "global"),
        a1: S.ptr(vdtype, "global"),
        a2: S.ptr(vdtype, "global"),
        a3: S.ptr(vdtype, "global"),
        c: S.ptr(dtype, "global"),
    ):
        S.vstore(S.vconcat([a0[0], a1[0], a2[0], a3[0]], part), c)

    return locals()[f"vconcat{inp_num}_func"]


def get_vconcat_gt(inps, part):
    half = len(inps[0]) // 2
    if part == "low":
        return np.concatenate([x[:half] for x in inps])
    elif part == "high":
        return np.concatenate([x[half:] for x in inps])
    elif part == "even":
        return np.concatenate([x[::2] for x in inps])
    elif part == "odd":
        return np.concatenate([x[1::2] for x in inps])
    else:
        assert False, f'Unsupported part "{part}"'


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("part", ("low", "high", "even", "odd"))
def test_vconcat2(dtype, part):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a0 = rand(n, dtype)
    a1 = rand(n, dtype)
    gt_out = get_vconcat_gt((a0, a1), part)

    prim_func = gen_vconcat_func(part, dtype, vdtype, 2)
    bm = aipu.tir.BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n, dtype)
    prim_func(a0, a1, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype)
    ex(a0, a1, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


@pytest.mark.parametrize("dtype", ("int8", "float16", "float32"))
@pytest.mark.parametrize("part", ("low", "high", "even", "odd"))
def test_vconcat3(dtype, part):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a0 = rand(n, dtype)
    a1 = rand(n, dtype)
    a2 = rand(n, dtype)
    gt_out = get_vconcat_gt((a0, a1, a2), part)

    prim_func = gen_vconcat_func(part, dtype, vdtype, 3)
    bm = aipu.tir.BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n * 3 // 2, dtype)
    prim_func(a0, a1, a2, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n * 3 // 2, dtype)
    ex(a0, a1, a2, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


@pytest.mark.parametrize("dtype", ("int8", "uint16", "int32"))
@pytest.mark.parametrize("part", ("low", "high", "even", "odd"))
def test_vconcat4(dtype, part):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a0 = rand(n, dtype)
    a1 = rand(n, dtype)
    a2 = rand(n, dtype)
    a3 = rand(n, dtype)
    gt_out = get_vconcat_gt((a0, a1, a2, a3), part)

    prim_func = gen_vconcat_func(part, dtype, vdtype, 4)
    bm = aipu.tir.BuildManager()
    ex = bm.build(prim_func)

    py_out = np.empty(n * 4 // 2, dtype)
    prim_func(a0, a1, a2, a3, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n * 4 // 2, dtype)
    ex(a0, a1, a2, a3, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_vconcat2("int8", part="low")
    test_vconcat2("int8", part="high")
    test_vconcat2("int8", part="even")
    test_vconcat2("int8", part="odd")
    test_vconcat3("int8", part="low")
    test_vconcat3("int8", part="high")
    test_vconcat3("int8", part="even")
    test_vconcat3("int8", part="odd")
    test_vconcat4("int8", part="low")
    test_vconcat4("int8", part="high")
    test_vconcat4("int8", part="even")
    test_vconcat4("int8", part="odd")
