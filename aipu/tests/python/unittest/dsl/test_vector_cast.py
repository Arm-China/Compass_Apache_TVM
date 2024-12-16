# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu, get_range, DataType
from tvm.aipu import script as S, testing
from tvm.aipu.utils import hw_native_vdtype, rand


def get_vector_cast_gt_out(from_vdtype, to_vdtype, saturate, a):
    if to_vdtype.is_float:
        return a.astype(to_vdtype.element_of)

    # From here, the target data type must be integer.
    if from_vdtype.is_integer:
        if saturate is True:
            a = np.clip(a, *get_range(to_vdtype))
        return a.astype(to_vdtype.element_of)

    # From here, the source data type must be float and the target data type must be integer.
    a = np.round(a)
    a = np.where(np.isnan(a), 0, a)
    # Here will promote to "float64" automatically, so it's safe.
    a = np.clip(a, *get_range("int32"))
    return a.astype(to_vdtype.element_of)


def is_unsupported_direct_cast(from_dtype, to_dtype):
    unsupported_direct_cast = (("uint32", "float32"), ("float32", "uint32"), ("uint32", "float16"))

    if (
        (from_dtype.is_float16 and to_dtype.is_integer and to_dtype.bits < 32)
        or (from_dtype.is_integer and from_dtype.bits < 32 and to_dtype.is_float16)
        or (str(from_dtype), str(to_dtype)) in unsupported_direct_cast
    ):
        return True

    return False


def is_invalid_saturate(from_dtype, to_dtype, saturate):
    if from_dtype.is_integer and to_dtype.is_integer:
        from_min, from_max = get_range(from_dtype)
        to_min, to_max = get_range(to_dtype)
        if to_min <= from_min and from_max <= to_max:
            return saturate is not None
        return False

    if to_dtype.is_float and saturate is not None:
        return True

    if from_dtype.is_float and to_dtype.is_integer:
        if to_dtype.is_int32:
            return saturate is False
        return saturate is not None

    return False


def is_invalid_combination0(from_dtype, to_dtype, saturate):
    from_dtype, to_dtype = DataType(from_dtype), DataType(to_dtype)

    if (
        from_dtype.bits <= to_dtype.bits
        or is_unsupported_direct_cast(from_dtype, to_dtype)
        or is_invalid_saturate(from_dtype, to_dtype, saturate)
    ):
        return True
    return False


def gen_cast_to_narrower_with_merge(from_vdtype, to_vdtype, saturate):
    @S.prim_func
    def cast_to_narrower_with_merge2(a: S.ptr(from_vdtype, "global"), out: S.ptr(to_vdtype, "global")):
        out[0] = S.cast((a[0], a[1]), to_vdtype, saturate=saturate)
        out[1] = S.cast((a[2], a[3]), to_vdtype, saturate=saturate)
        out[2] = S.cast((a[4], a[5]), to_vdtype, saturate=saturate)

    @S.prim_func
    def cast_to_narrower_with_merge4(a: S.ptr(from_vdtype, "global"), out: S.ptr(to_vdtype, "global")):
        out[0] = S.cast((a[0], a[1], a[2], a[3]), to_vdtype, saturate=saturate)

        cur_out = (out + 1).as_ptr(to_vdtype.element_of)
        va0 = S.cast((a[4], a[5], a[6]), to_vdtype, saturate=saturate)
        S.vstore(va0, cur_out, mask="24T8F")

        cur_out = cur_out + 24
        va1 = S.cast((a[7], a[8], a[9]), to_vdtype, saturate=saturate)
        S.vstore(va1, cur_out, mask="24T8F")

        cur_out = cur_out + 24
        va2 = S.cast((a[10], a[11]), to_vdtype, saturate=saturate)
        S.vstore(va2, cur_out, mask="16T16F")

    if from_vdtype.bits // to_vdtype.bits == 2:
        return cast_to_narrower_with_merge2
    return cast_to_narrower_with_merge4


@pytest.mark.parametrize("saturate", (None, True, False))
@pytest.mark.parametrize("to_dtype", ("int8", "uint8", "int16", "uint16", "float16"))
@pytest.mark.parametrize("from_dtype", ("int16", "uint16", "int32", "uint32", "float32"))
def test_cast_to_narrower_with_merge(from_dtype, to_dtype, saturate):
    if is_invalid_combination0(from_dtype, to_dtype, saturate):
        pytest.skip("Invalid combination.")

    from_vdtype, to_vdtype = hw_native_vdtype(from_dtype), hw_native_vdtype(to_dtype)
    n = to_vdtype.lanes * 3
    a = rand(n, from_dtype)
    gt_out = get_vector_cast_gt_out(from_vdtype, to_vdtype, saturate, a)

    py_func = gen_cast_to_narrower_with_merge(from_vdtype, to_vdtype, saturate)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=to_dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=to_dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def is_invalid_combination1(from_dtype, to_dtype, saturate):
    from_dtype, to_dtype = DataType(from_dtype), DataType(to_dtype)

    if (
        from_dtype.bits != to_dtype.bits
        or is_unsupported_direct_cast(from_dtype, to_dtype)
        or is_invalid_saturate(from_dtype, to_dtype, saturate)
    ):
        return True
    return False


def gen_cast_func(from_vdtype, to_vdtype, part, saturate):
    @S.prim_func
    def cast_func(a: S.ptr(from_vdtype, "global"), out: S.ptr(to_vdtype, "global")):
        out[0] = S.cast(a[0], to_vdtype, part, saturate=saturate)

    return cast_func


@pytest.mark.parametrize("saturate", (None, True, False))
@pytest.mark.parametrize("to_dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float32"))
@pytest.mark.parametrize("from_dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float32"))
def test_cast_to_same_bits(from_dtype, to_dtype, saturate):
    if is_invalid_combination1(from_dtype, to_dtype, saturate):
        pytest.skip("Invalid combination.")

    from_vdtype, to_vdtype = hw_native_vdtype(from_dtype), hw_native_vdtype(to_dtype)
    n = from_vdtype.lanes
    a = rand(n, from_dtype)
    gt_out = get_vector_cast_gt_out(from_vdtype, to_vdtype, saturate, a)

    py_func = gen_cast_func(from_vdtype, to_vdtype, "all", saturate)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=to_dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=to_dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def is_invalid_combination2(from_dtype, to_dtype, saturate):
    from_dtype, to_dtype = DataType(from_dtype), DataType(to_dtype)

    if (
        from_dtype.bits >= to_dtype.bits
        or is_unsupported_direct_cast(from_dtype, to_dtype)
        or is_invalid_saturate(from_dtype, to_dtype, saturate)
    ):
        return True
    return False


def gen_cast_to_wider_bits(from_vdtype, to_vdtype, saturate):
    @S.prim_func
    def cast_to_wider_bits2(a: S.ptr(from_vdtype, "global"), out: S.ptr(to_vdtype, "global")):
        out[0] = S.cast(a[0], to_vdtype, part="low", saturate=saturate)
        out[1] = S.cast(a[0], to_vdtype, part="high", saturate=saturate)
        a1_even = S.cast(a[1], to_vdtype, part="even", saturate=saturate)
        a1_odd = S.cast(a[1], to_vdtype, part="odd", saturate=saturate)
        out[2] = S.vzip(a1_even, a1_odd, part="low")
        out[3] = S.vzip(a1_even, a1_odd, part="high")

    @S.prim_func
    def cast_to_wider_bits4(a: S.ptr(from_vdtype, "global"), out: S.ptr(to_vdtype, "global")):
        out[0] = S.cast(a[0], to_vdtype, part="ll", saturate=saturate)
        out[1] = S.cast(a[0], to_vdtype, part="lh", saturate=saturate)
        out[2] = S.cast(a[0], to_vdtype, part="hl", saturate=saturate)
        out[3] = S.cast(a[0], to_vdtype, part="hh", saturate=saturate)

        out[4] = S.cast(a[1], to_vdtype, part="ll", saturate=saturate)
        out[5] = S.cast(a[1], to_vdtype, part="lh", saturate=saturate)
        out[6] = S.cast(a[1], to_vdtype, part="hl", saturate=saturate)
        out[7] = S.cast(a[1], to_vdtype, part="hh", saturate=saturate)

    return cast_to_wider_bits2 if to_vdtype.bits // from_vdtype.bits == 2 else cast_to_wider_bits4


@pytest.mark.parametrize("saturate", (None, True, False))
@pytest.mark.parametrize("to_dtype", ("int16", "uint16", "int32", "uint32", "float16", "float32"))
@pytest.mark.parametrize("from_dtype", ("int8", "uint8", "int16", "uint16", "float16"))
def test_cast_to_wider_bits(from_dtype, to_dtype, saturate):
    if is_invalid_combination2(from_dtype, to_dtype, saturate):
        pytest.skip("Invalid combination.")

    from_vdtype, to_vdtype = hw_native_vdtype(from_dtype), hw_native_vdtype(to_dtype)
    n = from_vdtype.lanes * 2
    a = rand(n, from_dtype)
    gt_out = get_vector_cast_gt_out(from_vdtype, to_vdtype, saturate, a)

    py_func = gen_cast_to_wider_bits(from_vdtype, to_vdtype, saturate)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=to_dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=to_dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


@pytest.mark.parametrize("saturate", (None, True, False))
@pytest.mark.parametrize("to_dtype", ("int8", "uint8", "int16", "uint16", "float16"))
@pytest.mark.parametrize("from_dtype", ("int16", "uint16", "int32", "uint32", "float32"))
def test_cast_to_narrower_bits(from_dtype, to_dtype, saturate):
    if is_invalid_combination0(from_dtype, to_dtype, saturate):
        pytest.skip("Invalid combination.")

    from_vdtype, to_vdtype = hw_native_vdtype(from_dtype), hw_native_vdtype(to_dtype)
    from_n = from_vdtype.lanes
    to_n = to_vdtype.lanes
    a = rand(from_n, from_dtype)
    gt_out = np.resize(get_vector_cast_gt_out(from_vdtype, to_vdtype, saturate, a), to_n)

    py_func = gen_cast_func(from_vdtype, to_vdtype, "all", saturate)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(to_n, dtype=to_dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out[:from_n], gt_out[:from_n])

    aipu_out = np.empty(to_n, dtype=to_dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out[:from_n], gt_out[:from_n])


if __name__ == "__main__":
    test_cast_to_narrower_with_merge("int32", "int8", True)
    test_cast_to_narrower_with_merge("float32", "uint8", None)
    test_cast_to_narrower_with_merge("int32", "float16", None)
    test_cast_to_same_bits("uint8", "uint8", None)
    test_cast_to_same_bits("int32", "uint32", True)
    test_cast_to_wider_bits("uint8", "int32", None)
    test_cast_to_wider_bits("uint8", "float32", None)
    test_cast_to_wider_bits("float16", "float32", None)
    test_cast_to_wider_bits("float16", "int32", True)
    test_cast_to_wider_bits("int16", "uint32", False)
    test_cast_to_narrower_bits("float32", "uint16", None)
    test_cast_to_narrower_bits("int32", "int8", True)
    test_cast_to_narrower_bits("float32", "float16", None)
