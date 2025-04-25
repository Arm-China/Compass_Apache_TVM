# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu, DataType, get_range
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_cast_func(from_dtype, to_dtype, hw_lanes):
    lanes0 = hw_lanes - 5
    lanes1 = hw_lanes + 3
    lanes2 = 2 * hw_lanes + 1
    lanes3 = 4 * hw_lanes

    @S.prim_func
    def cast_func(a: S.ptr(from_dtype, "global"), out: S.ptr(to_dtype, "global")):
        cur_a, cur_out = a, out
        # 1. n - 5, e.g., i32x3 -> fp32x3.
        va0 = S.vload(cur_a, lanes=lanes0)
        S.vstore(S.cast(va0, to_dtype), cur_out)

        cur_a, cur_out = cur_a + lanes0, cur_out + lanes0
        # 2. n + 3, e.g., i32x11 -> fp32x11.
        va1 = S.vload(cur_a, lanes=lanes1)
        S.vstore(S.cast(va1, to_dtype), cur_out)

        cur_a, cur_out = cur_a + lanes1, cur_out + lanes1
        # 3. 2 * n + 1, e.g., i32x17 -> fp32x17.
        va2 = S.vload(cur_a, lanes=lanes2)
        S.vstore(S.cast(va2, to_dtype), cur_out)

        cur_a, cur_out = cur_a + lanes2, cur_out + lanes2
        # 4. 4 * n, e.g., i32x32 -> fp32x32.
        va3 = S.vload(cur_a, lanes=lanes3)
        S.vstore(S.cast(va3, to_dtype), cur_out)

    return cast_func


def is_unsupported_direct_cast(from_dtype, to_dtype):
    from_dtype, to_dtype = DataType(from_dtype), DataType(to_dtype)
    unsupported_direct_cast = (("uint32", "float32"), ("float32", "uint32"), ("uint32", "float16"))

    if (
        (from_dtype.is_float16 and to_dtype.is_integer and to_dtype.bits < 32)
        or (from_dtype.is_integer and from_dtype.bits < 32 and to_dtype.is_float16)
        or (str(from_dtype), str(to_dtype)) in unsupported_direct_cast
    ):
        return True

    return False


def get_vector_cast_gt_out(from_dtype_str, to_dtype_str, a):
    from_dtype = DataType(from_dtype_str)
    to_dtype = DataType(to_dtype_str)

    if from_dtype_str == "uint32" and to_dtype.is_float16:
        a = a.astype("int32")

    if from_dtype.is_float and to_dtype.is_integer:
        a = np.round(a)

    if from_dtype.is_float32 and to_dtype.is_integer:
        # Here will promote to "float64" automatically, so it's safe.
        a = np.clip(a, *get_range("int32"))
    return a.astype(to_dtype_str)


@pytest.mark.parametrize("to_dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("from_dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_vector_cast(from_dtype, to_dtype):
    if is_unsupported_direct_cast(from_dtype, to_dtype):
        pytest.skip("Invalid combination.")

    from_vdtype, to_vdtype = hw_native_vdtype(from_dtype), hw_native_vdtype(to_dtype)
    hw_lanes = min(from_vdtype.lanes, to_vdtype.lanes)
    n = 8 * hw_lanes - 1
    a = rand(n, from_dtype)
    gt_out = get_vector_cast_gt_out(from_dtype, to_dtype, a)

    py_func = gen_cast_func(from_dtype, to_dtype, hw_lanes)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=to_dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=to_dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def gen_cast_literal_func(n, dtype):
    const_vector = list(range(n))

    @S.prim_func
    def cast_literal_func(a: S.ptr(dtype, "global"), out: S.ptr(dtype, "global")):
        S.vstore(S.cast(const_vector, dtype), out)

    return cast_literal_func


@pytest.mark.parametrize("n", (3, 9, 27, 33))
def test_vector_cast_literal(n):
    dtype = "float32"
    a = rand(n, dtype)
    gt_out = np.array(list(range(n)), dtype)

    py_func = gen_cast_literal_func(n, dtype)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_vector_cast("float32", "int32")
    test_vector_cast("float16", "uint32")
    test_vector_cast("float16", "float32")
    test_vector_cast("float32", "float16")
    test_vector_cast("int8", "int32")
    test_vector_cast("int32", "uint8")
    test_vector_cast("uint16", "float32")
    test_vector_cast_literal(33)
