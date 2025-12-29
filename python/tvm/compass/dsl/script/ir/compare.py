# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""The compare part of IR APIs."""
import builtins
import numpy as np
from tvm import tir, DataType
from ....compass_info import CompassInfo
from ..pysim import PyVar, binary_op_match_types
from .base import register_ir_api, canonicalize_mask
from .utils import PARAM_R_MARK, broadcast_scalar, assert_vdtype_match, canonicalize_r
from .utils import is_scalar, is_vector, assert_neither_flexible_nor_multiple_width_vector
from .utils import assert_not_flexible_width_vector, assert_not_bfloat16_scalar
from .bitwise import vand


@register_ir_api
def vceq(x, y, mask=None):
    """Check if active elements of ``x`` are equal to the corresponding elements of ``y``.

    - The inactive elements of result vector are set to False.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: 1  1  3  4  5  6   7   8
           y: 1  1  3  5  5  5  99  99
        mask: T  F  T  T  T  F   T   T

         out = S.vceq(x, y, mask)
         out: T  F  T  F  T  F   F   F

    Parameters
    ----------
    x, y : Union[PrimExpr, int, float]
        The operands. If either one is a scalar, it will be automatically broadcast.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32", "bfloat16".

    Examples
    --------
    .. code-block:: python

        mask_out = S.vceq(va, vb)
        mask_out = S.vceq(va, 3)
        mask_out = S.vceq(3, vb)
        mask_out = S.vceq(va, vb, mask="3T5F")
        mask_out = S.vceq(va, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vceq, __vqeq
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return tir.call_extern(mask.dtype, "vceq", x, y, mask)


@register_ir_api
def _py_vceq(x, y, mask=None):
    x, y = broadcast_scalar(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(x == y, DataType(f"boolx{x.dtype.lanes}"), mask, False)


@register_ir_api
def vcneq(x, y, mask=None):
    """Check if active elements of ``x`` are not equal to the corresponding elements of ``y``.

    - The inactive elements of result vector are set to False.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: 9  9  9  9  9  9  99  99
           y: 1  1  9  9  2  3  99  99
        mask: T  F  T  T  T  T   T   F

         out = S.vcneq(x, y, mask)
         out: T  F  F  F  T  T   F   F

    Parameters
    ----------
    x, y : Union[PrimExpr, int, float]
        The operands. If either one is a scalar, it will be automatically broadcast.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32", "bfloat16".

    Examples
    --------
    .. code-block:: python

        mask_out = S.vcneq(va, vb)
        mask_out = S.vcneq(va, 3)
        mask_out = S.vcneq(3, vb)
        mask_out = S.vcneq(va, vb, mask="3T5F")
        mask_out = S.vcneq(va, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vcneq, __vqne
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return tir.call_extern(mask.dtype, "vcneq", x, y, mask)


@register_ir_api
def _py_vcneq(x, y, mask=None):
    x, y = broadcast_scalar(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(x != y, DataType(f"boolx{x.dtype.lanes}"), mask, False)


@register_ir_api
def vcge(x, y, mask=None):
    """Check if active elements of ``x`` are greater than or equal to the corresponding elements of
    ``y``.

    - The inactive elements of result vector are set to False.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: 1  1  2  3  4  5  5  6
           y: 5  5  5  5  5  5  5  5
        mask: T  F  T  T  T  T  F  T

         out = S.vcge(x, y, mask)
         out: F  F  F  F  F  T  F  T

    Parameters
    ----------
    x, y : Union[PrimExpr, int, float]
        The operands. If either one is a scalar, it will be automatically broadcast.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32", "bfloat16".

    Examples
    --------
    .. code-block:: python

        mask_out = S.vcge(va, vb)
        mask_out = S.vcge(va, 3)
        mask_out = S.vcge(3, vb)
        mask_out = S.vcge(va, vb, mask="3T5F")
        mask_out = S.vcge(va, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vcge, __vqge
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return tir.call_extern(mask.dtype, "vcge", x, y, mask)


@register_ir_api
def _py_vcge(x, y, mask=None):
    x, y = broadcast_scalar(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(x >= y, DataType(f"boolx{x.dtype.lanes}"), mask, False)


@register_ir_api
def vcgt(x, y, mask=None):
    """Check if active elements of ``x`` are greater than the corresponding elements of ``y``.

    - The inactive elements of result vector are set to False.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: 1  1  2  3  4  5  6  6
           y: 1  1  3  3  3  3  3  3
        mask: T  F  T  T  T  T  T  F

         out = S.vcgt(x, y, mask)
         out: F  F  F  F  T  T  T  F

    Parameters
    ----------
    x, y : Union[PrimExpr, int, float]
        The operands. If either one is a scalar, it will be automatically broadcast.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32", "bfloat16".

    Examples
    --------
    .. code-block:: python

        mask_out = S.vcgt(va, vb)
        mask_out = S.vcgt(va, 3)
        mask_out = S.vcgt(3, vb)
        mask_out = S.vcgt(va, vb, mask="3T5F")
        mask_out = S.vcgt(va, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vcgt, __vqgt
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return tir.call_extern(mask.dtype, "vcgt", x, y, mask)


@register_ir_api
def _py_vcgt(x, y, mask=None):
    x, y = broadcast_scalar(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(x > y, DataType(f"boolx{x.dtype.lanes}"), mask, False)


@register_ir_api
def vcle(x, y, mask=None):
    """Check if active elements of ``x`` are less than or equal to the corresponding elements of
    ``y``.

    - The inactive elements of result vector are set to False.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: 1  1  2  3  4  5  6  6
           y: 1  1  3  3  3  3  3  3
        mask: T  F  T  T  T  T  T  F

         out = S.vcle(x, y, mask)
         out: T  F  T  T  F  F  F  F

    Parameters
    ----------
    x, y : Union[PrimExpr, int, float]
        The operands. If either one is a scalar, it will be automatically broadcast.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32", "bfloat16".

    Examples
    --------
    .. code-block:: python

        mask_out = S.vcle(va, vb)
        mask_out = S.vcle(va, 3)
        mask_out = S.vcle(3, vb)
        mask_out = S.vcle(va, vb, mask="3T5F")
        mask_out = S.vcle(va, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vcle, __vqle
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return tir.call_extern(mask.dtype, "vcle", x, y, mask)


@register_ir_api
def _py_vcle(x, y, mask=None):
    x, y = broadcast_scalar(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(x <= y, DataType(f"boolx{x.dtype.lanes}"), mask, False)


@register_ir_api
def vclt(x, y, mask=None):
    """Check if active elements of ``x`` are less than the corresponding elements of ``y``.

    - The inactive elements of result vector are set to False.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: 1  1  2  3  4  5  6  3
           y: 1  1  4  4  4  4  4  4
        mask: T  F  T  T  T  T  T  F

         out = S.vclt(x, y, mask)
         out: F  F  T  T  F  F  F  F

    Parameters
    ----------
    x, y : Union[PrimExpr, int, float]
        The operands. If either one is a scalar, it will be automatically broadcast.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32", "bfloat16".

    Examples
    --------
    .. code-block:: python

        mask_out = S.vclt(va, vb)
        mask_out = S.vclt(va, 3)
        mask_out = S.vclt(3, vb)
        mask_out = S.vclt(va, vb, mask="3T5F")
        mask_out = S.vclt(va, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vclt, __vqlt
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return tir.call_extern(mask.dtype, "vclt", x, y, mask)


@register_ir_api
def _py_vclt(x, y, mask=None):
    x, y = broadcast_scalar(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(x < y, DataType(f"boolx{x.dtype.lanes}"), mask, False)


@register_ir_api
def isnan(x, mask=None):
    """Check if active elements of ``x`` are ``NaN``.

    - The scalar situation where ``x`` is a scalar is also supported.
    - The inactive elements of result vector are set to False.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: nan  1.0  nan  3.6  nan  5.0  6.7  3.1
        mask:  T    T    F    T    T    T    F    T

         out = S.isnan(x, mask)
         out:  T    F    F    F    T    F    F    F

    Parameters
    ----------
    x : Union[PrimExpr, float]
        The operand.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "float16/32", "bfloat16".

    Examples
    --------
    .. code-block:: python

        bool_out = S.isnan(scalar_a)
        mask_out = S.isnan(vector_b, mask)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __class, __vclass, __vqne, __vand, isnan
    """
    assert isinstance(x, tir.PrimExpr), "The 1st arg expect a variable."
    x_dtype = x.dtype
    msg = f'The data type of the 1st arg expect floating, but got: "{x.dtype}".'
    assert x_dtype.is_floating, msg

    if is_scalar(x):
        assert_not_bfloat16_scalar(x_dtype)
        ret = tir.call_extern("int32", "__class", x)
        ret = ret & 0x3  # signalingNaN: 0x1, quietNaN: 0x2.
        return ret != 0

    mask = canonicalize_mask(mask, x_dtype.lanes)
    ret = tir.call_extern(x_dtype.with_int(), "__vclass", PARAM_R_MARK, x, mask)
    ret = vand(ret, 0x3, mask)
    return vcneq(ret, 0, mask)


@register_ir_api
def _py_isnan(x, mask=None):
    if is_scalar(x):
        return bool(np.isnan(x))

    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(np.isnan(x), DataType(f"boolx{x.dtype.lanes}"), mask, False)


@register_ir_api
def isinf(x, mask=None):
    """Check if active elements of ``x`` are positive or negative infinity.

    - The scalar situation where ``x`` is a scalar is also supported.
    - The inactive elements of result vector are set to False.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: inf  1.0  -inf  3.1  inf  5.0  6.3  -inf
        mask:  T    T     F    T    T    T    F     T

         out = S.isinf(x, mask)
         out:  T    F     F    F    T    F    F     T

    Parameters
    ----------
    x : Union[PrimExpr, float]
        The operand.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "float16/32", "bfloat16".

    Examples
    --------
    .. code-block:: python

        bool_out = S.isinf(scalar_a)
        mask_out = S.isinf(vector_b, mask)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __class, __vclass, __vqne, __vand, isinf
    """
    assert isinstance(x, tir.PrimExpr), "The 1st arg expect a variable."
    x_dtype = x.dtype
    msg = f'The data type of the 1st arg expect floating, but got: "{x.dtype}".'
    assert x_dtype.is_floating, msg

    if is_scalar(x):
        assert_not_bfloat16_scalar(x_dtype)
        ret = tir.call_extern("int32", "__class", x)
        ret = ret & 0x204  # positiveInfinity: 0x200, negativeInfinity: 0x4.
        return ret != 0

    mask = canonicalize_mask(mask, x_dtype.lanes)
    ret = tir.call_extern(x_dtype.with_int(), "__vclass", PARAM_R_MARK, x, mask)
    ret = vand(ret, 0x204, mask)
    return vcneq(ret, 0, mask)


@register_ir_api
def _py_isinf(x, mask=None):
    if is_scalar(x):
        return bool(np.isinf(x))

    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(np.isinf(x), DataType(f"boolx{x.dtype.lanes}"), mask, False)


@register_ir_api
def isfinite(x, mask=None):
    """Check if active elements of ``x`` are finite number.

    - The scalar situation where ``x`` is a scalar is also supported.
    - The inactive elements of result vector are set to False.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: inf  1.0  -inf  3.1  inf  5.0  6.3  nan
        mask:  T    T     F    T    T    T    F    T

         out = S.isfinite(x, mask)
         out:  F    T     F    T    F    T    F    F

    Parameters
    ----------
    x : Union[PrimExpr, float]
        The operand.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "float16/32", "bfloat16".

    Examples
    --------
    .. code-block:: python

        bool_out = S.isfinite(scalar_a)
        mask_out = S.isfinite(vector_b, mask)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __class, __vclass, __vqne, __vand, isfinite
    """
    assert isinstance(x, tir.PrimExpr), "The 1st arg expect a variable."
    x_dtype = x.dtype
    msg = f'The data type of the 1st arg expect floating, but got: "{x.dtype}".'
    assert x_dtype.is_floating, msg

    if is_scalar(x):
        assert_not_bfloat16_scalar(x_dtype)
        ret = tir.call_extern("int32", "__class", x)
        # negativeNormal: 0x8
        # negativeSubnormal: 0x10
        # negativeZero: 0x20
        # positiveZero: 0x40
        # positiveSubnormal: 0x80
        # positiveNormal: 0x100
        ret = ret & 0x1F8
        return ret != 0

    mask = canonicalize_mask(mask, x_dtype.lanes)
    ret = tir.call_extern(x_dtype.with_int(), "__vclass", PARAM_R_MARK, x, mask)
    ret = vand(ret, 0x1F8, mask)
    return vcneq(ret, 0, mask)


@register_ir_api
def _py_isfinite(x, mask=None):
    if is_scalar(x):
        return bool(np.isfinite(x))

    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(np.isfinite(x), DataType(f"boolx{x.dtype.lanes}"), mask, False)


def _vmax(x, y, mask=None, r=None):
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    vdtype = x.dtype
    r = canonicalize_r(r, vdtype)
    mask = canonicalize_mask(None if r == PARAM_R_MARK else mask, vdtype.lanes)
    return tir.call_extern(vdtype, "__vmax", r, x, y, mask)


def _py_vmax(x, y, mask=None, r=None):
    x, y = broadcast_scalar(x, y)
    mask = canonicalize_mask(None if r is None else mask, x.dtype.lanes)
    return PyVar(np.maximum(x, y), x.dtype, mask, r)


@register_ir_api
def max(x, y, mask=None, r=None):  # pylint: disable=redefined-builtin
    """Computes the maximum on active elements of ``x`` with the corresponding elements of ``y``.

    - The scalar situation where both ``x`` and ``y`` are scalar is also supported.
    - The inactive elements of result vector are determined by ``r``.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: 1  2  3  4  5  6  7  8
           y: 5  5  5  5  5  5  5  5
        mask: T  T  T  T  F  F  T  T
           z: 9  8  7  6  4  3  2  1

         out = S.max(x, y, mask, r=z)
         out: 5  5  5  5  4  3  7  8

    Parameters
    ----------
    x, y : Union[PrimExpr, int, float]
        The operands. If either one is a scalar, it will be automatically broadcast.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    r : Optional[PrimExpr, int, float]
        Provide the value of the inactive elements in result vector. If it is a scalar, it will be
        automatically broadcast. ``None`` means the inactive elements of result vector are
        undefined.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32", "bfloat16".

    Examples
    --------
    .. code-block:: python

         c = S.max(3, 4)
        vc = S.max(va, vb)
        vc = S.max(va, 3)
        vc = S.max(3, vb)
        vc = S.max(va, vb, mask="3T5F")
        vc = S.max(va, vb, mask=S.tail_mask(n, 8))
        vc = S.max(va, vb, mask="T7F", r=vb)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vmax
    """
    if is_scalar(x) and is_scalar(y):
        x, y = binary_op_match_types(x, y)
        return tir.max(x, y)
    return _vmax(x, y, mask, r)


@register_ir_api
def _py_max(x, y, mask=None, r=None):
    if is_scalar(x) and is_scalar(y):
        x, y = binary_op_match_types(x, y)
        return PyVar(np.max([x, y]), DataType(x.dtype))
    return _py_vmax(x, y, mask, r)


def _vmin(x, y, mask=None, r=None):
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    vdtype = x.dtype
    r = canonicalize_r(r, vdtype)
    mask = canonicalize_mask(None if r == PARAM_R_MARK else mask, vdtype.lanes)
    return tir.call_extern(vdtype, "__vmin", r, x, y, mask)


def _py_vmin(x, y, mask=None, r=None):
    x, y = broadcast_scalar(x, y)
    mask = canonicalize_mask(None if r is None else mask, x.dtype.lanes)
    return PyVar(np.minimum(x, y), x.dtype, mask, r)


@register_ir_api
def min(x, y, mask=None, r=None):  # pylint: disable=redefined-builtin
    """Computes the minimum on active elements of ``x`` with the corresponding elements of ``y``.

    - The scalar situation where both ``x`` and ``y`` are scalar is also supported.
    - The inactive elements of result vector are determined by ``r``.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: 1  2  3  4  5  6  7  8
           y: 5  5  5  5  5  5  5  5
        mask: T  T  T  T  F  F  T  T
           z: 9  8  7  6  4  3  2  1

         out = S.min(x, y, mask, r=z)
         out: 1  2  3  4  4  3  5  5

    Parameters
    ----------
    x, y : Union[PrimExpr, int, float]
        The operands. If either one is a scalar, it will be automatically broadcast.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    r : Optional[PrimExpr, int, float]
        Provide the value of the inactive elements in result vector. If it is a scalar, it will be
        automatically broadcast. ``None`` means the inactive elements of result vector are
        undefined.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32", "bfloat16".

    Examples
    --------
    .. code-block:: python

         c = S.min(3, 4)
        vc = S.min(va, vb)
        vc = S.min(va, 3)
        vc = S.min(3, vb)
        vc = S.min(va, vb, mask="3T5F")
        vc = S.min(va, vb, mask=S.tail_mask(n, 8))
        vc = S.min(va, vb, mask="T7F", r=vb)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vmin
    """
    if is_scalar(x) and is_scalar(y):
        x, y = binary_op_match_types(x, y)
        return tir.min(x, y)
    return _vmin(x, y, mask, r)


@register_ir_api
def _py_min(x, y, mask=None, r=None):
    if is_scalar(x) and is_scalar(y):
        x, y = binary_op_match_types(x, y)
        return PyVar(np.min([x, y]), DataType(x.dtype))
    return _py_vmin(x, y, mask, r)


@register_ir_api
def vmaxh(x, y):
    """Computes the maximum of two adjacent elements, the results of ``x`` are placed to the low
    half part of result vector, and that of ``y`` are placed to the high half part.

    - The feature Multiple Width Vector is supported.

    .. code-block::

          x: 9  1  8  2  7  3  6  4
          y: 5  4  4  3  3  2  2  1

        out = S.vmaxh(x, y)
        out: 9  8  7  6  5  4  3  2

    Parameters
    ----------
    x, y : Union[PrimExpr, int]
        The operands. If either one is a scalar, it will be automatically broadcast.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vmaxh(va, vb)
        vc = S.vmaxh(va, 3)
        vc = S.vmaxh(3, vb)
        vc = S.vmaxh(va, vb, mask="3T5F")
        vc = S.vmaxh(va, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vmaxh
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    vdtype = x.dtype
    assert_not_flexible_width_vector(vdtype)
    assert vdtype.is_integer, "Only support integer instruction."
    return tir.call_extern(vdtype, "__vmaxh", x, y)


@register_ir_api
def _py_vmaxh(x, y):
    x, y = broadcast_scalar(x, y)
    half_lanes = x.dtype.lanes // 2

    ret = PyVar.zeros(x.dtype)
    for i in range(half_lanes):
        ret[i] = builtins.max(x[2 * i], x[2 * i + 1])
        ret[i + half_lanes] = builtins.max(y[2 * i], y[2 * i + 1])
    return ret


@register_ir_api
def vminh(x, y):
    """Computes the minimum of two adjacent elements, the results of ``x`` are placed to the low
    half part of result vector, and that of ``y`` are placed to the high half part.

    - The feature Multiple Width Vector is supported.

    .. code-block::

          x: 9  1  8  2  7  3  6  4
          y: 5  4  4  3  3  2  2  1

        out = S.vminh(x, y)
        out: 1  2  3  4  4  3  2  1

    Parameters
    ----------
    x, y : Union[PrimExpr, int]
        The operands. If either one is a scalar, it will be automatically broadcast.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vminh(va, vb)
        vc = S.vminh(va, 3)
        vc = S.vminh(3, vb)
        vc = S.vminh(va, vb, mask="3T5F")
        vc = S.vminh(va, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vminh
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    vdtype = x.dtype
    assert_not_flexible_width_vector(vdtype)
    assert vdtype.is_integer, "Only support integer instruction."
    return tir.call_extern(vdtype, "__vminh", x, y)


@register_ir_api
def _py_vminh(x, y):
    x, y = broadcast_scalar(x, y)
    half_lanes = x.dtype.lanes // 2

    ret = PyVar.zeros(x.dtype)
    for i in range(half_lanes):
        ret[i] = builtins.min(x[2 * i], x[2 * i + 1])
        ret[i + half_lanes] = builtins.min(y[2 * i], y[2 * i + 1])
    return ret


@register_ir_api
def vrpmax(x, mask=None, return_idx=False):
    """Computes the reduction maximum of all active elements of ``x``, and places the result as the
    lowest elements of result vector.

    - The remaining upper elements of result vector are undefined.

    .. code-block::

           x: 7  9  3  4  5  6  7  8
        mask: T  F  T  T  T  T  T  T

         out = S.vrpmax(x, mask)
         out: 8  ?  ?  ?  ?  ?  ?  ?

         out = S.vrpmax(x, mask, return_idx=True)
         out: 8  7  ?  ?  ?  ?  ?  ?

    Parameters
    ----------
    x : PrimExpr
        The operands.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    return_idx : Optional[bool]
        If True, it will record the index of the maximum value and put it in the second element of
        result vector. If there are more than one maximum value, the index will be the lowest one.
        This argument is only supported by the V3 architecture.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32", "bfloat16".

    Examples
    --------
    .. code-block::

        vc = S.vrpmax(va)
        vc = S.vrpmax(va,  S.tail_mask(4, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vrpmax
    """
    if return_idx:
        cps_info = CompassInfo.current()
        msg = f'The arg "return_idx" does not support the target "{cps_info.name}".'
        assert cps_info.version != "X2", msg

    assert is_vector(x), "The 1st arg expect a vector."
    vdtype = x.dtype
    assert_neither_flexible_nor_multiple_width_vector(vdtype)

    mask = canonicalize_mask(mask, vdtype.lanes)
    func_name = "__vrpmaxe" if return_idx else "__vrpmax"
    return tir.call_extern(vdtype, func_name, PARAM_R_MARK, x, mask)


@register_ir_api
def _py_vrpmax(x, mask=None, return_idx=False):
    mask = canonicalize_mask(mask, x.dtype.lanes)
    active_indices = np.where(mask)[0]
    active_values = x[active_indices]

    ret = PyVar.zeros(x.dtype)
    ret[0] = np.max(active_values)
    if return_idx:
        idx = active_indices[np.where(active_values == ret[0])[0][0]]
        ret[1] = np.array([idx]).view(x.dtype.element_of)[0]
    return ret


@register_ir_api
def vrpmin(x, mask=None, return_idx=False):
    """Computes the reduction minimum of all active elements of ``x``, and places the result as the
    lowest elements of result vector.

    - The remaining upper elements of result vector are undefined.

    .. code-block::

           x: 7  9  3  4  5  6  7  8
        mask: T  T  F  T  T  T  T  T

         out = S.vrpmin(x, mask)
         out: 4  ?  ?  ?  ?  ?  ?  ?

         out = S.vrpmin(x, mask, return_idx=True)
         out: 4  3  ?  ?  ?  ?  ?  ?

    Parameters
    ----------
    x : PrimExpr
        The operands.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    return_idx : Optional[bool]
        If True, it will record the index of the minimum value and put it in the second element of
        result vector. If there are more than one minimum value, the index will be the lowest one.
        This argument is only supported by the V3 architecture.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32", "bfloat16".

    Examples
    --------
    .. code-block:: python

        vc = S.vrpmin(va)
        vc = S.vrpmin(va,  S.tail_mask(4, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vrpmin
    """
    if return_idx:
        cps_info = CompassInfo.current()
        msg = f'The arg "return_idx" does not support the target "{cps_info.name}".'
        assert cps_info.version != "X2", msg

    assert is_vector(x), "The 1st arg expect a vector."
    vdtype = x.dtype
    assert_neither_flexible_nor_multiple_width_vector(vdtype)

    mask = canonicalize_mask(mask, vdtype.lanes)
    func_name = "__vrpmine" if return_idx else "__vrpmin"
    return tir.call_extern(vdtype, func_name, PARAM_R_MARK, x, mask)


@register_ir_api
def _py_vrpmin(x, mask=None, return_idx=False):
    mask = canonicalize_mask(mask, x.dtype.lanes)
    active_indices = np.where(mask)[0]
    active_values = x[active_indices]

    ret = PyVar.zeros(x.dtype)
    ret[0] = np.min(active_values)
    if return_idx:
        idx = active_indices[np.where(active_values == ret[0])[0][0]]
        ret[1] = np.array([idx]).view(x.dtype.element_of)[0]
    return ret


__all__ = (
    "vceq",
    "vcneq",
    "vcge",
    "vcgt",
    "vcle",
    "vclt",
    "isnan",
    "isinf",
    "isfinite",
    "max",
    "min",
    "vmaxh",
    "vminh",
    "vrpmax",
    "vrpmin",
)
