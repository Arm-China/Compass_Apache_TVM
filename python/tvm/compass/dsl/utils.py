# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Common Compass DSL utilities."""
import os
from ... import tir, ir, DataType, int_within_range


HW_NATIVE_STORAGE_DTYPES = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16")
HW_NATIVE_STORAGE_DTYPES += ("float32", "bfloat16")
HW_NATIVE_SCALAR_DTYPES = HW_NATIVE_STORAGE_DTYPES + ("bool",)
HW_NATIVE_VDTYPES = ("int8x32", "uint8x32", "int16x16", "uint16x16", "int32x8", "uint32x8")
HW_NATIVE_VDTYPES += ("float16x16", "float32x8", "bfloat16x16")
HW_NATIVE_MASK_TYPES = ("boolx8", "boolx16", "boolx32")


def is_hw_native_scalar_dtype(dtype):
    return dtype in HW_NATIVE_SCALAR_DTYPES


def is_hw_native_dtype(dtype):
    return dtype in (HW_NATIVE_SCALAR_DTYPES + HW_NATIVE_VDTYPES + HW_NATIVE_MASK_TYPES)


def is_hw_native_vdtype(dtype):
    return dtype in HW_NATIVE_VDTYPES


VALID_PTR_ELEMENT_DTYPES = HW_NATIVE_STORAGE_DTYPES + ("void",)
ALIAS2ELEMENT_DTYPE = {
    "i8": "int8",
    "u8": "uint8",
    "i16": "int16",
    "u16": "uint16",
    "i32": "int32",
    "u32": "uint32",
    "fp16": "float16",
    "fp32": "float32",
    "bf16": "bfloat16",
}


def resolve_dtype_alias(dtype):
    """Replace the alias in the given data type to the corresponding complete form."""
    if isinstance(dtype, DataType):
        return dtype

    for k, v in ALIAS2ELEMENT_DTYPE.items():
        dtype = dtype.replace(k, v)
    return DataType(dtype)


def abspath(path, base_dir=None):
    """Return the absolute path of the given path and the base directory.

    Parameters
    ----------
    path : Optional[str]
        The given path.

    base_dir : Optional[str]
        The base directory will be used only when the given path is a relative
        one, if it is None, the current working directory will be used.

    Return
    ------
    ret : Optional[str]
        The result absolute path. None if the given path is None.
    """
    if path is None:
        return None

    path = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(path):
        # "abspath" here is used to remove "." and "..".
        path = os.path.abspath(path)
    else:
        path = os.path.abspath(f"{base_dir or os.getcwd()}/{path}")
    return path


def hw_native_vdtype(dtype):
    """Get the corresponding hardware native vector data type.

    Parameters
    ----------
    dtype : Union[str, DataType]
        The given data type, can be any scalar or vector data type except boolean ones.

    Returns
    -------
    ret: DataType
        The corresponding hardware native vector data type.

    Examples
    --------
    .. code-block:: python

        # Generate from string objects.
        i8x32 = hw_native_vdtype("int8")
        fp32x8 = hw_native_vdtype("float32")
        fp16x16 = hw_native_vdtype("fp16")

        # Generate from DataType objects.
        u16x16 = hw_native_vdtype(DataType("uint16"))
        i32x8 = hw_native_vdtype(DataType("int32"))

    See Also
    --------
    - :doc:`../../language_basics/types`
    """
    scalar_dtype = resolve_dtype_alias(dtype)
    assert not scalar_dtype.is_bool, "Does not support boolean data type."
    return scalar_dtype.with_lanes(256 // scalar_dtype.bits)


def double_elem_width(vdtype, allow_64bit=False):
    if not allow_64bit and vdtype.bits == 32:
        return vdtype

    new_bits = vdtype.bits * 2
    # For the type u8x8, the result type should be u16x8 instead of u16x4.
    new_lanes = vdtype.lanes
    if new_bits * new_lanes > 256:
        new_lanes //= 2
    return vdtype.with_bits(new_bits).with_lanes(new_lanes)


def half_elem_width(vdtype, double_lanes=True):
    assert vdtype.bits >= 8

    # For the type u16x8, maybe the expect result type is u8x8 instead of u8x16.
    new_lanes = vdtype.lanes * 2 if double_lanes else vdtype.lanes
    return vdtype.with_bits(vdtype.bits // 2).with_lanes(new_lanes)


def get_binary_op_result_type(ltype_or_lhs, rtype_or_rhs):
    """Infer the binary operation result type through the same logic of C++ "BinaryOpMatchTypes", in
    addition, adjusting integer literal type is considered."""
    ltype_or_lhs = DataType(ltype_or_lhs) if isinstance(ltype_or_lhs, str) else ltype_or_lhs
    rtype_or_rhs = DataType(rtype_or_rhs) if isinstance(rtype_or_rhs, str) else rtype_or_rhs
    ltype_or_lhs = DataType("float32") if isinstance(ltype_or_lhs, float) else ltype_or_lhs
    rtype_or_rhs = DataType("float32") if isinstance(rtype_or_rhs, float) else rtype_or_rhs
    ltype_or_lhs = DataType("bool") if isinstance(ltype_or_lhs, bool) else ltype_or_lhs
    rtype_or_rhs = DataType("bool") if isinstance(rtype_or_rhs, bool) else rtype_or_rhs
    assert all(isinstance(x, (DataType, int)) for x in (ltype_or_lhs, rtype_or_rhs))

    if all(isinstance(x, int) for x in (ltype_or_lhs, rtype_or_rhs)):
        return DataType("int32")

    ltype, rtype = ltype_or_lhs, rtype_or_rhs
    # Only handle the situation that all operands are integers. For the situation that all operands
    # are floating, because can't know whether a float literal can be represented by float16 or not,
    # so can't do this. For other situations, "binary_op_match_types" is good enough.
    if isinstance(ltype_or_lhs, int):
        if rtype.is_integer and int_within_range(ltype_or_lhs, rtype):
            return rtype.element_of
        ltype = DataType("int32")

    if isinstance(rtype_or_rhs, int):
        if ltype.is_integer and int_within_range(rtype_or_rhs, ltype):
            return ltype.element_of
        rtype = DataType("int32")

    ltype, rtype = ltype.element_of, rtype.element_of

    if ltype == rtype:
        return ltype

    # Promote to higher bits, e.g., i8 + i16 -> i16 + i16, fp16 + fp32 -> fp32 + fp32.
    if (
        (ltype.is_float and rtype.is_float)
        or (ltype.is_bfloat and rtype.is_bfloat)
        or (ltype.is_int and rtype.is_int)
        or (ltype.is_uint and rtype.is_uint)
    ):
        return ltype if ltype.bits > rtype.bits else rtype

    # Cast x -> float when the other operand is float.
    if ltype.is_float or rtype.is_float:
        return ltype if ltype.is_float else rtype

    # Cast x -> bfloat when the other operand is bfloat.
    if ltype.is_bfloat or rtype.is_bfloat:
        return ltype if ltype.is_bfloat else rtype

    # Handle mixing signed and unsigned integers.
    assert (ltype.is_int and rtype.is_uint) or (ltype.is_uint and rtype.is_int)

    if ltype.bits > rtype.bits:
        return ltype
    if ltype.bits < rtype.bits:
        return rtype

    # The width of signed and unsigned integers is same.
    return ltype if ltype.is_uint else rtype


def is_pointer(x):
    return isinstance(x, tir.Call) and x.op == ir.Op.get("tir.pointer")


def is_type_annotation(x):
    return isinstance(x, tir.Call) and x.op == ir.Op.get("tir.type_annotation")
