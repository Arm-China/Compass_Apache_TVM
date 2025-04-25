# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Utilities for IR API implementation."""
import numpy as np
from tvm import tir, DataType, get_range
from ...utils import is_hw_native_scalar_dtype, HW_NATIVE_VDTYPES, HW_NATIVE_MASK_TYPES
from ..pysim import PyVar, PySimInfo


PARAM_R_MARK = tir.StringImm("ParamR")
VALID_PARTS = ("all", "low", "high", "even", "odd", "ll", "lh", "hl", "hh")
VALID_ATTRS = {"no_zip": ("vcast",)}


def is_hw_native_vdtype_or_mask_type(dtype):
    return str(dtype) in (HW_NATIVE_VDTYPES + HW_NATIVE_MASK_TYPES)


def assert_neither_flexible_nor_multiple_width_vector(dtype):
    msg = f'This API expect hardware native vector types, but got: "{dtype}".'
    assert DataType(dtype).is_scalar or is_hw_native_vdtype_or_mask_type(dtype), msg


def assert_not_flexible_width_vector(dtype):
    dtype = DataType(dtype)
    assert dtype.is_scalar or (
        is_hw_native_scalar_dtype(dtype.element_of)
        and dtype.lanes != 0
        and dtype.total_bits % (8 if dtype.is_bool else 256) == 0
    ), f'This API does not support the feature Flexible Width Vector, but got: "{dtype}".'


# For PySim and IR Builder, a scalar literal is represented by Python built-in type.
def _is_float_scalar_const(x):
    return isinstance(x, float) or (isinstance(x, str) and x.lower() in ("inf", "-inf", "nan"))


def is_scalar_const(x):
    return isinstance(x, int) or _is_float_scalar_const(x)


# For IR Builder, a variable (i.e., primitive expression) is represented by class "tir.PrimExpr".
def is_scalar_prim_expr(x):
    return isinstance(x, tir.PrimExpr) and DataType(x.dtype).is_scalar


def is_integer_scalar(x):
    return isinstance(x, int) or (
        isinstance(x, tir.PrimExpr) and DataType(x.dtype).is_integer_scalar
    )


def is_float_scalar(x):
    return _is_float_scalar_const(x) or (
        isinstance(x, tir.PrimExpr) and DataType(x.dtype).is_float_scalar
    )


def is_scalar(x):
    if PySimInfo.current is None:
        assert not isinstance(x, tir.Pointer), f'Unexpect "{x.dtype}*" pointer as argument.'
        return is_integer_scalar(x) or is_float_scalar(x)

    # Implementation of PySim.
    # For PySim, a variable (i.e., primitive expression) is represented by class "PyVar".
    return isinstance(x, (int, float)) or (isinstance(x, PyVar) and x.dtype.is_scalar)


def is_vector_or_mask(x):
    return isinstance(x, (PyVar, tir.PrimExpr)) and DataType(x.dtype).is_vector


def is_vector(x):
    return is_vector_or_mask(x) and not DataType(x.dtype).is_bool


def get_dtype(x):
    if isinstance(x, int):
        ret = "int32"
    elif isinstance(x, float):
        ret = "float32"
    elif isinstance(x, np.ndarray) and x.ndim == 1:
        ret = f"{x.dtype}x{len(x)}"
    else:
        ret = x.dtype
    return DataType(ret)


def within_range(value, dtype=None, low=None, high=None):
    """Check whether the value is in the given range or not."""
    if low is None or high is None:
        assert dtype is not None, 'The arg "dtype" must be given, when "low" or "high" isn\'t set.'

    if dtype is not None:
        dtype = DataType(dtype)
        min_value, max_value = get_range(dtype)
        if get_dtype(value).is_float and not dtype.is_float:
            return False

    low = min_value if low is None else low
    if high is None:
        if dtype.is_float:
            high = (np.float64(max_value).view("int64") + 1).view("float64")
        else:
            high = max_value + 1
    return low <= value < high


def assert_vdtype_match(*args, ignore_sign=False):
    """Ensure the type of all of the vector arguments is matched."""
    if len(args) <= 1:
        return

    assert all(is_vector_or_mask(x) for x in args), "All arguments must be vector or mask."

    first_vdtype = None
    for i, arg in enumerate(args):
        cur_vdtype = DataType(arg.dtype)

        if first_vdtype is None:
            first_vdtype = cur_vdtype
            continue

        if cur_vdtype.is_integer and first_vdtype.is_integer:
            if cur_vdtype.with_int() != first_vdtype.with_int():
                msg = f'Argument type mismatch: 0-th: "{first_vdtype}" vs. {i}-th: "{cur_vdtype}".'
                raise ValueError(msg)
            if ignore_sign is False and cur_vdtype.type_code != first_vdtype.type_code:
                msg = 'The sign of operands is different, if the API have parameter "out_sign" '
                msg += "please specify the output sign."
                raise ValueError(msg)
        elif cur_vdtype != first_vdtype:
            msg = f'Argument type mismatch: 0-th: "{first_vdtype}" vs. {i}-th: "{cur_vdtype}".'
            raise ValueError(msg)


def assert_lanes_valid(lanes):
    """Check the parameter "lanes"."""
    assert isinstance(lanes, int), 'The arg "lanes" expect an integer scalar constant.'
    assert lanes > 1, f'The arg "lanes" expect > 1, but got: "{lanes}".'


def _get_candidate_vdtypes(args):
    vector_args = tuple(x for x in args if is_vector_or_mask(x))
    if PySimInfo.current is None:
        assert_vdtype_match(*vector_args, ignore_sign=True)

    vdtype = DataType(vector_args[0].dtype)
    ret = [vdtype]
    if not vdtype.is_bool and vdtype.is_integer:
        ret.append(vdtype.with_uint() if vdtype.is_int else vdtype.with_int())
    return tuple(ret)


def can_safe_cast(x, dtype):
    """Check if the given expression or value can be cast to the specified data type safely."""
    to_dtype = DataType(dtype)

    if is_scalar_const(x):
        return within_range(x, to_dtype)

    from_dtype = DataType(x.dtype)
    if from_dtype.is_float and to_dtype.is_integer:
        return False

    from_min, from_max = get_range(from_dtype)
    to_min, to_max = get_range(to_dtype)
    return to_min <= from_min and from_max <= to_max


def _get_most_fit_vdtype(x, candidate_vdtypes):
    for vdtype in candidate_vdtypes:
        if can_safe_cast(x, vdtype):
            return vdtype

    value_or_dtype = x if is_scalar_const(x) else x.dtype
    raise ValueError(f'Can\'t broadcast "{value_or_dtype}" to any of "{candidate_vdtypes}".')


def broadcast_scalar(*args):
    """Broadcast the scalar in the given arguments automatically."""
    from .conversion import cast  # pylint: disable=import-outside-toplevel

    scalar_args = tuple((i, x) for i, x in enumerate(args) if is_scalar(x))

    if PySimInfo.current is None:
        assert len(args) > 1
        assert len(scalar_args) != len(args), "All arguments are scalar."

    if len(scalar_args) == 0:
        return args

    candidate_vdtypes = _get_candidate_vdtypes(args)

    ret = list(args)
    for i, x in scalar_args:
        ret[i] = cast(x, _get_most_fit_vdtype(x, candidate_vdtypes))

    return ret


def change_sign_if_needed(dtype, sign):
    """Change the sign part of the data type with the given sign value."""
    assert sign in (None, "u", "s"), f"Invalid value: {sign}."

    dtype = DataType(dtype)
    if sign is None or not dtype.is_integer:
        return dtype
    if sign == "u":
        return dtype.with_uint()
    return dtype.with_int()


def canonicalize_r(r, ret_vdtype):
    """Canonicalize the special parameter "r" with the return data type."""
    from .conversion import cast  # pylint: disable=import-outside-toplevel

    if r in (None, PARAM_R_MARK):
        return PARAM_R_MARK

    r_dtype = get_dtype(r)
    if r_dtype == ret_vdtype:
        return r

    if r_dtype.element_of == ret_vdtype.element_of:  # Only lanes is different.
        return cast(r, ret_vdtype)

    # The lanes and element dtype of r and the return data type are all different.
    msg = f'Argument type mismatch: r: "{r_dtype}" vs. return data type: "{ret_vdtype}".'
    assert not (r_dtype.is_float and ret_vdtype.is_integer), msg

    # Collect all the compatible data types. For integer ret_vdtype, includes equal-bit integer
    # types signed or not, check if the parameter r can be canonicalized through cast.
    compatible_vdtypes = [ret_vdtype]
    if r_dtype.is_integer and ret_vdtype.is_integer:
        diff_sign_vdtype = ret_vdtype.with_uint() if ret_vdtype.is_int else ret_vdtype.with_int()
        compatible_vdtypes.append(diff_sign_vdtype)

    if is_scalar_const(r):
        msg1 = f'Argument r "{r}" is out of range of return data type "{ret_vdtype}".'
        assert any(within_range(r, vdtype) for vdtype in compatible_vdtypes), msg1
        return cast(r, ret_vdtype)

    # For expression, check if the parameter r is compatible with the return data type.
    r_min, r_max = get_range(r_dtype)
    for vdtype in compatible_vdtypes:
        ret_min, ret_max = get_range(vdtype)
        if ret_min <= r_min and r_max <= ret_max:
            break
    else:  # The parameter r can't fall in the range of any compatible data type.
        assert False, msg
    return cast(r, ret_vdtype)
