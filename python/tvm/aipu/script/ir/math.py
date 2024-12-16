# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The math part of IR APIs."""
import numpy as np
from tvm import tir, DataType
from ..pysim import PyVar, binary_op_match_types
from .base import register_ir_api
from .utils import broadcast_scalar, assert_vdtype_match, is_scalar, get_dtype, is_integer_scalar


@register_ir_api
def exp(x):
    """Computes the exponential of ``x``.

    - The scalar situation where ``x`` is a scalar is also supported.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    x : Union[PrimExpr, float]
        The operand.

    Returns
    -------
    ret : PrimExpr
        The result.

    Supported DType
    ---------------
        "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.exp(va)
        scalar_b = S.exp(1.23456)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: exp
    """
    dtype = get_dtype(x)
    assert dtype.is_float, "Only support float input."
    return tir.call_extern(dtype, "exp" if dtype.is_scalar else "vexp", x)


@register_ir_api
def _py_exp(x):
    return PyVar(np.exp(x), get_dtype(x))


@register_ir_api
def log(x):
    """Computes the natural logarithm of ``x``.

    - The scalar situation where ``x`` is a scalar is also supported.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    x : Union[PrimExpr, float]
        The operands.

    Returns
    -------
    ret : PrimExpr
        The result.

    Supported DType
    ---------------
        "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.log(va)
        scalar_b = S.log(1.23456)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: log
    """
    dtype = get_dtype(x)
    assert DataType(dtype).is_float, "Only support float input."
    return tir.call_extern(dtype, "log" if dtype.is_scalar else "vlog", x)


@register_ir_api
def _py_log(x):
    return PyVar(np.log(x), get_dtype(x))


@register_ir_api
def tanh(x):
    """Computes the hyperbolic tangent of ``x``.

    - The scalar situation where ``x`` is a scalar is also supported.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    x : Union[PrimExpr, float]
        The operands.

    Returns
    -------
    ret : PrimExpr
        The result.

    Supported DType
    ---------------
        "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.tanh(va)
        scalar_b = S.tanh(1.23456)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: tanh
    """
    dtype = get_dtype(x)
    assert dtype.is_float, "Only support float input."
    return tir.call_extern(dtype, "tanh" if dtype.is_scalar else "vtanh", x)


@register_ir_api
def _py_tanh(x):
    return PyVar(np.tanh(x), get_dtype(x))


@register_ir_api
def sin(x):
    """Compute the trigonometric sine of ``x``.

    - The scalar situation where ``x`` is a scalar is also supported.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    x : Union[PrimExpr, float]
        The operands.

    Returns
    -------
    ret : PrimExpr
        The result.

    Supported DType
    ---------------
        "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.sin(va)
        scalar_b = S.sin(1.23456)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: sin
    """
    dtype = get_dtype(x)
    assert dtype.is_float, "Only support float input."
    return tir.call_extern(dtype, "sin" if dtype.is_scalar else "vsin", x)


@register_ir_api
def _py_sin(x):
    return PyVar(np.sin(x), get_dtype(x))


@register_ir_api
def cos(x):
    """Computes the trigonometric cosine of ``x``.

    - The scalar situation where ``x`` is a scalar is also supported.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    x : Union[PrimExpr, float]
        The operands.

    Returns
    -------
    ret : PrimExpr
        The result.

    Supported DType
    ---------------
        "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.cos(va)
        scalar_b = S.cos(1.23456)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: cos
    """
    dtype = get_dtype(x)
    assert dtype.is_float, "Only support float input."
    return tir.call_extern(dtype, "cos" if dtype.is_scalar else "vcos", x)


@register_ir_api
def _py_cos(x):
    return PyVar(np.cos(x), get_dtype(x))


@register_ir_api
def rsqrt(x):
    """Computes the reciprocal of the square-root of ``x``.

    - The scalar situation where ``x`` is a scalar is also supported.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    x : Union[PrimExpr, float]
        The operands.

    Returns
    -------
    ret : PrimExpr
        The result.

    Supported DType
    ---------------
        "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.rsqrt(va)
        scalar_b = S.rsqrt(1.23456)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: rsqrt
    """
    dtype = get_dtype(x)
    assert dtype.is_float, "Only support float input."
    return tir.call_extern(dtype, "rsqrt" if dtype.is_scalar else "vrsqrt", x)


@register_ir_api
def _py_rsqrt(x):
    dtype = get_dtype(x).element_of
    one = getattr(np, dtype)(1)
    neg_zero = getattr(np, dtype)(-0.0)
    mask = np.logical_or(np.isneginf(x), x == neg_zero)

    # According to experiment:
    # 1. Negative values -0.0, -inf, obeys formula: "1 / sqrt(x)".
    # 2. Other normal floating, may exist 1 bit error. Obeys: "sqrt(1 / x)".
    ret = np.where(mask, one / np.sqrt(x), np.sqrt(one / x))
    return PyVar(ret, dtype)


@register_ir_api
def sqrt(x):
    """Computes the square-root of ``x``.

    - The scalar situation where ``x`` is a scalar is also supported.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: 1  4  9  16 25 36 49 64

         out = S.sqrt(x)
         out: 1  2  3  4  5  6  7  8

    Parameters
    ----------
    x : Union[PrimExpr, float]
        The operands.

    Returns
    -------
    ret : PrimExpr
        The result.

    Supported DType
    ---------------
        "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.sqrt(va)
        scalar_b = S.sqrt(1.23456)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: sqrt
    """
    dtype = get_dtype(x)
    assert dtype.is_float, "Only support float input."
    return tir.call_extern(dtype, "sqrt" if dtype.is_scalar else "vsqrt", x)


@register_ir_api
def _py_sqrt(x):
    return PyVar(np.sqrt(x), get_dtype(x))


@register_ir_api
def floor(x):
    """Computes the floor of ``x``.

    - The scalar situation where ``x`` is a scalar is also supported.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    x : Union[PrimExpr, float]
        The operands.

    Returns
    -------
    ret : PrimExpr
        The result.

    Supported DType
    ---------------
        "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.floor(va)
        scalar_b = S.floor(1.23456)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: floor
    """
    dtype = get_dtype(x)
    assert dtype.is_float, "Only support float input."
    return tir.call_extern(dtype, "floor" if dtype.is_scalar else "vfloor", x)


@register_ir_api
def _py_floor(x):
    return PyVar(np.floor(x), get_dtype(x))


@register_ir_api
def ceil(x):
    """Computes the ceil of ``x``.

    - The scalar situation where ``x`` is a scalar is also supported.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    x : Union[PrimExpr, float]
        The operands.

    Returns
    -------
    ret : PrimExpr
        The result.

    Supported DType
    ---------------
        "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.ceil(va)
        scalar_b = S.ceil(1.23456)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: ceil
    """
    dtype = get_dtype(x)
    assert dtype.is_float, "Only support float input."
    return tir.call_extern(dtype, "ceil" if dtype.is_scalar else "vceil", x)


@register_ir_api
def _py_ceil(x):
    return PyVar(np.ceil(x), get_dtype(x))


@register_ir_api
def ceildiv(x, y):
    """Computes the ceil division on integer scalar ``x`` and ``y``.

    Parameters
    ----------
    x, y : Union[PrimExpr, int]
        The operands. They must be integer scalar.

    Returns
    -------
    ret : PrimExpr
        The result.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32".

    Examples
    --------
    .. code-block:: python

        b = S.ceildiv(scalar_a, 8)
        b = S.ceildiv(lsram_a[0], 8)  # The "lsram_a" is a scalar buffer.

    """
    assert is_integer_scalar(x), 'The arg "x" expect an integer scalar.'
    assert is_integer_scalar(y), 'The arg "y" expect an integer scalar.'

    x, y = binary_op_match_types(x, y)
    return tir.call_extern(x.dtype, "ceildiv", x, y)


@register_ir_api
def _py_ceildiv(x, y):
    if y == 0:
        raise ZeroDivisionError("The 2nd arg can't be zero.")

    x, y = binary_op_match_types(x, y)
    one = np.ones_like(x)  # Assign type to avoid unexpected type promotion.
    return PyVar(np.trunc((x + y - one) / y), x.dtype)


@register_ir_api
def pow(x, exponent):  # pylint: disable=redefined-builtin
    """Computes the power of ``x`` with the corresponding elements of ``exponent``.

    - The scalar situation where both ``x`` and ``exponent`` are scalar is also supported.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    x : Union[PrimExpr, float]
        The base value. If it is a scalar in the vector situation, it will be automatically
        broadcast.

    exponent : Union[PrimExpr, float]
        The exponent value. If it is a scalar in the vector situation, it will be automatically
        broadcast.

    Returns
    -------
    ret : PrimExpr
        The result.

    Supported DType
    ---------------
        "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.pow(va, vb)
        scalar_b = S.pow(2.0, 3.0)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: pow
    """
    if is_scalar(x) and is_scalar(exponent):
        x, exponent = binary_op_match_types(x, exponent)
    else:
        x, exponent = broadcast_scalar(x, exponent)
        assert_vdtype_match(x, exponent)

    dtype = DataType(x.dtype)
    assert dtype.is_float, "Only support float instruction."
    return tir.call_extern(dtype, "pow" if dtype.is_scalar else "vpow", x, exponent)


@register_ir_api
def _py_pow(x, exponent):
    if is_scalar(x) and is_scalar(exponent):
        x, exponent = binary_op_match_types(x, exponent)
    else:
        x, exponent = broadcast_scalar(x, exponent)
    return PyVar(np.power(x, exponent), x.dtype)


__all__ = (
    "exp",
    "log",
    "tanh",
    "sin",
    "cos",
    "rsqrt",
    "sqrt",
    "floor",
    "ceil",
    "ceildiv",
    "pow",
)
