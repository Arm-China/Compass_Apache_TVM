# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""The arithmetic part of IR APIs."""
import numpy as np
from tvm import tir, DataType, get_range
from ....compass_info import CompassInfo
from ...utils import double_elem_width, half_elem_width, hw_native_vdtype
from ..pysim import PyVar, binary_op_match_types, pysim_run_sim
from .base import register_ir_api, canonicalize_mask
from .utils import PARAM_R_MARK, broadcast_scalar, assert_vdtype_match, change_sign_if_needed
from .utils import is_scalar, is_vector, canonicalize_r, assert_not_flexible_width_vector
from .utils import assert_neither_flexible_nor_multiple_width_vector, is_scalar_const, can_safe_cast
from .utils import get_dtype, assert_not_bfloat16_scalar


@register_ir_api
def vadd(x, y, mask=None, saturate=False, out_sign=None, r=None):
    """Computes the addition on active elements of ``x`` with the corresponding elements of ``y``.

    - The inactive elements of result vector are determined by ``r``.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: 1  2  3  4  5  6   7   8
           y: 1  2  3  4  5  6   7   8
        mask: T  T  T  T  F  F   T   T
           z: 9  8  7  6  4  3   2   1

         out = S.vadd(x, y, mask, r=z)
         out: 2  4  6  8  4  3  14  16

    Parameters
    ----------
    x, y : Union[PrimExpr, int, float]
        The operands. If either one is a scalar, it will be automatically broadcast.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    saturate : Optional[bool]
        Whether the result needs to be saturated or not.

    out_sign : Optional[str]
        Specify whether the output sign is signed or unsigned. It is only needed for integer
        operation. ``None`` means same as operands, so the sign of operands must be the same, ``u``
        means unsigned, ``s`` means signed.

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

        vc = S.vadd(va, vb)
        vc = S.vadd(va, 3)
        vc = S.vadd(va, vb, saturate=True)
        vc = S.vadd(va, vb, out_sign="u")
        vc = S.vadd(va, vb, mask=S.tail_mask(n, 8))
        vc = S.vadd(va, vb, mask="3T5F", r=vb)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vadd, __vadds
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y, ignore_sign=(out_sign is not None))
    x_vdtype = x.dtype
    assert not (x_vdtype.is_floating and saturate), "Saturate is meaningless for floating."
    ret_vdtype = change_sign_if_needed(x_vdtype, out_sign)
    r = canonicalize_r(r, ret_vdtype)

    mask = canonicalize_mask(None if r == PARAM_R_MARK else mask, x_vdtype.lanes)
    return tir.call_extern(ret_vdtype, "vadd", r, x, y, mask, saturate)


@register_ir_api
def _py_vadd(x, y, mask=None, saturate=False, out_sign=None, r=None):
    x, y = broadcast_scalar(x, y)
    ret_vdtype = change_sign_if_needed(x.dtype, out_sign)

    if saturate:
        i64_x = x.astype("int64")
        i64_y = y.astype("int64")
        min_value, max_value = get_range(ret_vdtype)
        ret = np.clip(i64_x + i64_y, min_value, max_value)
    else:
        ret = x + y

    mask = canonicalize_mask(None if r is None else mask, x.dtype.lanes)
    return PyVar(ret, ret_vdtype, mask, r)


@register_ir_api
def vaddh(x, y):
    """Performs an add operation on every two adjacent elements in the vector x and vector y,
    concats the results of x and y, and places the results of x to the lower half part and the
    results of y to the higher half part.

    - The feature Multiple Width Vector is supported.

    .. code-block::

          x:  9   1   8   2   7   3  6  4
          y:  9   4   8   3   7   2  6  1

        out = S.vaddh(x, y)
        out: 10  10  10  10  13  11  9  7

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

        vc = S.vaddh(va, vb)
        vc = S.vaddh(va, 3)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vaddh
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    assert_not_flexible_width_vector(x.dtype)
    return tir.call_extern(x.dtype, "__vaddh", x, y)


@register_ir_api
def _py_vaddh(x, y):
    x, y = broadcast_scalar(x, y)
    dt_min, dt_max = get_range(x.dtype)
    n = x.dtype.lanes

    ret = PyVar.zeros(x.dtype)
    x, y = x.astype("int64"), y.astype("int64")
    for i in range(n // 2):
        ret[i] = np.clip(x[2 * i] + x[2 * i + 1], dt_min, dt_max)
        ret[i + n // 2] = np.clip(y[2 * i] + y[2 * i + 1], dt_min, dt_max)
    return ret


@register_ir_api
def abs(x, mask=None, saturate=False):  # pylint: disable=redefined-builtin
    """Computes the absolute value of x.

    - The scalar situation where ``x`` is a scalar is also supported.
    - The inactive elements of result vector are undefined.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: 1  -2  -3  4  -5  6  -127  -128
        mask: T   T   T  T   F  F    T     T

         out = S.abs(x, mask)
         out: 1   2   3  4   ?  ?   127  -128

         out = S.abs(x, mask, saturate=True)
         out: 1   2   3  4   ?  ?   127   127

    Parameters
    ----------
    x: Union[PrimExpr, int, float]
        The operands.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    saturate : Optional[bool]
        Whether the result needs to be saturated or not.

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

        vc = S.abs(va)
        vc = S.abs(va, saturate=True)
        vc = S.abs(va, mask="3T5F")
        vc = S.abs(va, mask=S.tail_mask(n, 8))
        scalar_c = S.abs(-10)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __abs, __vabs, __vabss
    """
    dtype = get_dtype(x)
    if dtype.is_scalar:
        assert_not_bfloat16_scalar(dtype)
        assert mask is None, "Only support mask in vector scenario."
        assert not (dtype.is_floating and saturate), "Saturate is meaningless for floating."
        assert not saturate, "Currently can't support saturate in scalar scenario."
        return tir.call_extern(dtype, "__abs", x)

    x_vdtype = x.dtype
    assert not (x_vdtype.is_floating and saturate), "Saturate is meaningless for floating."

    mask = canonicalize_mask(mask, x_vdtype.lanes)
    return tir.call_extern(x_vdtype, "vabs", PARAM_R_MARK, x, mask, saturate)


@register_ir_api
def _py_abs(x, mask=None, saturate=False):
    dtype = get_dtype(x)
    if dtype.is_scalar:
        return PyVar(np.abs(x), dtype)

    if saturate:
        ret = np.minimum(np.abs(x.astype("int64")), np.iinfo(x.dtype.element_of).max)
    else:
        ret = np.abs(x)

    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(ret, x.dtype, mask)


@register_ir_api
def vsub(x, y, mask=None, saturate=False, out_sign=None, r=None):
    """Computes the subtraction on active elements of ``x`` with the corresponding elements of
    ``y``.

    - The inactive elements of result vector are determined by ``r``.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: 1  2  3  4  5  6  7  8
           y: 1  1  1  1  2  2  2  2
        mask: T  T  T  T  F  F  T  T
           z: 9  8  7  6  4  3  2  1

         out = S.vsub(x, y, mask, r=z)
         out: 0  1  2  3  4  3  5  6

    Parameters
    ----------
    x, y : Union[PrimExpr, int, float]
        The operands. If either one is a scalar, it will be automatically broadcast.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    saturate : Optional[bool]
        Whether the result needs to be saturated or not.

    out_sign : Optional[str]
        Specify whether the output sign is signed or unsigned. It is only needed for integer
        operation. ``None`` means same as operands, so the sign of operands must be the same, ``u``
        means unsigned, ``s`` means signed.

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

        vc = S.vsub(va, vb)
        vc = S.vsub(va, 3)
        vc = S.vsub(va, vb, saturate=True)
        vc = S.vsub(va, vb, out_sign="u")
        vc = S.vsub(va, vb, mask=S.tail_mask(n, 8))
        vc = S.vsub(va, vb, mask="3T5F", r=vb)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vsub, __vsubs
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y, ignore_sign=(out_sign is not None))
    x_vdtype = x.dtype
    assert not (x_vdtype.is_floating and saturate), "Saturate is meaningless for floating."
    ret_vdtype = change_sign_if_needed(x_vdtype, out_sign)
    r = canonicalize_r(r, ret_vdtype)

    mask = canonicalize_mask(None if r == PARAM_R_MARK else mask, x_vdtype.lanes)
    return tir.call_extern(ret_vdtype, "vsub", r, x, y, mask, saturate)


@register_ir_api
def _py_vsub(x, y, mask=None, saturate=False, out_sign=None, r=None):
    x, y = broadcast_scalar(x, y)
    ret_vdtype = change_sign_if_needed(x.dtype, out_sign)

    if saturate:
        i64_x = x.astype("int64")
        i64_y = y.astype("int64")
        min_value, max_value = get_range(ret_vdtype)
        ret = np.clip(i64_x - i64_y, min_value, max_value)
    else:
        ret = x - y

    mask = canonicalize_mask(None if r is None else mask, x.dtype.lanes)
    return PyVar(ret, ret_vdtype, mask, r)


@register_ir_api
def vsubh(x, y):
    """Performs a sub operation on every two adjacent elements in the vector x and vector y, concats
    the results of x and y, and places the results of x to the lower half part and the results of y
    to the higher half part.

    - The feature Multiple Width Vector is supported.

    .. code-block::

          x: 9  1  8  2  7  3  6  4
          y: 9  4  8  3  7  2  6  1

        out = S.vsubh(x, y)
        out: 8  6  4  2  5  5  5  5

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

        vc = S.vsubh(va, vb)
        vc = S.vsubh(va, 3)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vsubh
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    assert_not_flexible_width_vector(x.dtype)
    return tir.call_extern(x.dtype, "__vsubh", x, y)


@register_ir_api
def _py_vsubh(x, y):
    x, y = broadcast_scalar(x, y)
    dt_min, dt_max = get_range(x.dtype)
    n = x.dtype.lanes

    ret = PyVar.zeros(x.dtype)
    x, y = x.astype("int64"), y.astype("int64")
    for i in range(n // 2):
        ret[i] = np.clip(x[2 * i] - x[2 * i + 1], dt_min, dt_max)
        ret[i + n // 2] = np.clip(y[2 * i] - y[2 * i + 1], dt_min, dt_max)
    return ret


@register_ir_api
def vmul(x, y, mask=None, out_sign=None, r=None):
    """Computes the multiplication on active elements of ``x`` with the corresponding elements of
    ``y``.

    - The inactive elements of result vector are determined by ``r``.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: 1  2  3   4  5  6   7   8
           y: 1  2  3   4  5  6   7   8
        mask: T  T  T   T  F  F   T   T
           z: 9  8  7   6  4  3   2   1

         out = S.vmul(x, y, mask, r=z)
         out: 1  4  9  16  4  3  49  64

    Parameters
    ----------
    x, y : Union[PrimExpr, int, float]
        The operands. If either one is a scalar, it will be automatically broadcast.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    out_sign : Optional[str]
        Specify whether the output sign is signed or unsigned. It is only needed for integer
        operation. ``None`` means same as operands, so the sign of operands must be the same, ``u``
        means unsigned, ``s`` means signed.

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
        "int16/32", "uint16/32", "float16/32", "bfloat16".

    Examples
    --------
    .. code-block:: python

        vc = S.vmul(va, vb)
        vc = S.vmul(va, 3)
        vc = S.vmul(va, vb, out_sign="u")
        vc = S.vmul(va, vb, mask="3T5F")
        vc = S.vmul(va, vb, mask=S.tail_mask(n, 8))
        vc = S.vmul(va, vb, mask"T7F", r=vb)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vmul, __vmulh
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y, ignore_sign=(out_sign is not None))
    x_vdtype = x.dtype
    assert x_vdtype.bits != 8, "The 8bit equal-width multiply is meaningless."
    ret_vdtype = change_sign_if_needed(x_vdtype, out_sign)
    r = canonicalize_r(r, ret_vdtype)

    mask = canonicalize_mask(None if r == PARAM_R_MARK else mask, x_vdtype.lanes)
    return tir.call_extern(ret_vdtype, "vmul", r, x, y, mask)


@register_ir_api
def _py_vmul(x, y, mask=None, out_sign=None, r=None):
    x, y = broadcast_scalar(x, y)
    ret_vdtype = change_sign_if_needed(x.dtype, out_sign)
    mask = canonicalize_mask(None if r is None else mask, x.dtype.lanes)
    return PyVar(x * y, ret_vdtype, mask, r)


@register_ir_api
def vmull(x, y, mask=None, out_sign=None, r=None):
    """Computes the multiplication on active elements of low half part of ``x`` with the
    corresponding elements of ``y``.
    Expands elements bit: 8bit -> 16bit or 16bit -> 32bit.

    - The inactive elements of result vector are determined by ``r``.

    .. code-block::

         x(i16x16): 1  2  3  4  5  6   7  8  9  0  1  2   3  4   5  6
         y(i16x16): 1  2  3  4  5  6   7  8  9  0  1  2   3  4   5  6
              mask: T  T  T  T  F  F   T  T  T  T  T  T   F  F   F  F
          z(i32x8): 9     8     7      6     5     4      3      2

        out = S.vmull(x, y, mask, r=z)
        out(i32x8): 1     4     9     16     5     4     49     64

    Parameters
    ----------
    x, y : Union[PrimExpr, int]
        The operands. If either one is a scalar, it will be automatically broadcast.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    out_sign : Optional[str]
        Specify whether the output sign is signed or unsigned. It is only needed for integer
        operation. ``None`` means same as operands, so the sign of operands must be the same, ``u``
        means unsigned, ``s`` means signed.

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
        "int8/16/32", "uint8/16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vmull(va, vb)
        vc = S.vmull(va, 3)
        vc = S.vmull(va, vb, out_sign="u")
        vc = S.vmull(va, vb, mask="3T5F")
        vc = S.vmull(va, vb, mask=S.tail_mask(n, 8))
        vc = S.vmull(va, vb, mask="T7F", r=vb)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vmul
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y, ignore_sign=(out_sign is not None))
    x_vdtype = x.dtype
    assert_neither_flexible_nor_multiple_width_vector(x_vdtype)
    assert x_vdtype.is_integer, "Only support integer instruction."

    ret_vdtype = change_sign_if_needed(double_elem_width(x_vdtype), out_sign)
    r = canonicalize_r(r, ret_vdtype)
    mask = canonicalize_mask(None if r == PARAM_R_MARK else mask, x_vdtype.lanes)
    return tir.call_extern(ret_vdtype, "vmull", r, x, y, mask)


@register_ir_api
def _py_vmull(x, y, mask=None, out_sign=None, r=None):
    x, y = broadcast_scalar(x, y)
    x_vdtype = x.dtype
    ret_vdtype = change_sign_if_needed(double_elem_width(x_vdtype, True), out_sign)

    data_cnt = x_vdtype.lanes // 2
    ret = x[:data_cnt].astype(ret_vdtype.element_of) * y[:data_cnt].astype(ret_vdtype.element_of)

    mask = canonicalize_mask(None if r is None else mask, x_vdtype.lanes)
    ret_mask = mask[:data_cnt]

    if x_vdtype.bits == 32:
        # The current result vector type is something like "uint64x4", need to be converted to
        # "uint32x8". The current return mask is something like "[0, 1, 0, 1]", need to be converted
        # to "[0, 0, 1, 1, 0, 0, 1, 1]".
        ret_vdtype = half_elem_width(ret_vdtype)
        ret = ret.view(ret_vdtype.element_of)
        ret_mask = ret_mask.repeat(2)

    return PyVar(ret, ret_vdtype, ret_mask, r)


@register_ir_api
def vmulh(x, y, mask=None, out_sign=None, r=None):
    """Computes the multiplication on active elements of high half part of ``x`` with the
    corresponding elements of ``y``.
    Expands elements bit: 8bit -> 16bit or 16bit -> 32bit.

    - The inactive elements of result vector are determined by ``r``.

    .. code-block::

         x(i16x16):  1  2  3  4  5  6  7  8  9  0  1  2  3  4  5  6
         y(i16x16):  1  2  3  4  5  6  7  8  9  0  1  2  3  4  5  6
              mask:  T  T  T  T  F  F  T  T  T  T  T  T  F  F  F  F
          z(i32x8):  9     8     7     6     5     4     3     2

        out = S.vmulh(x, y, mask, r=z)
        out(i32x8): 81     0     1     4     5     4     3     2

    Parameters
    ----------
    x, y : Union[PrimExpr, int]
        The operands. If either one is a scalar, it will be automatically broadcast.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    out_sign : Optional[str]
        Specify whether the output sign is signed or unsigned. It is only needed for integer
        operation. ``None`` means same as operands, so the sign of operands must be the same, ``u``
        means unsigned, ``s`` means signed.

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
        "int8/16/32", "uint8/16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vmulh(va, vb)
        vc = S.vmulh(va, 3)
        vc = S.vmulh(va, vb, out_sign="u")
        vc = S.vmulh(va, vb, mask="3T5F")
        vc = S.vmulh(va, vb, mask=S.tail_mask(n, 8))
        vc = S.vmulh(va, vb, mask="T7F", r=vb)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vmulh
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y, ignore_sign=(out_sign is not None))
    x_vdtype = x.dtype
    assert_neither_flexible_nor_multiple_width_vector(x_vdtype)
    assert x_vdtype.is_integer, "Only support integer instruction."

    ret_vdtype = change_sign_if_needed(double_elem_width(x_vdtype), out_sign)
    r = canonicalize_r(r, ret_vdtype)
    mask = canonicalize_mask(None if r == PARAM_R_MARK else mask, x_vdtype.lanes)
    return tir.call_extern(ret_vdtype, "vmulh", r, x, y, mask)


@register_ir_api
def _py_vmulh(x, y, mask=None, out_sign=None, r=None):
    x, y = broadcast_scalar(x, y)
    x_vdtype = x.dtype
    ret_vdtype = change_sign_if_needed(double_elem_width(x_vdtype, True), out_sign)

    data_cnt = x_vdtype.lanes // 2
    ret = x[data_cnt:].astype(ret_vdtype.element_of) * y[data_cnt:].astype(ret_vdtype.element_of)

    mask = canonicalize_mask(None if r is None else mask, x_vdtype.lanes)
    ret_mask = mask[data_cnt:]

    if x_vdtype.bits == 32:
        # The current result vector type is something like "uint64x4", need to be converted to
        # "uint32x8". The current return mask is something like "[0, 1, 0, 1]", need to be converted
        # to "[0, 0, 1, 1, 0, 0, 1, 1]".
        ret_vdtype = half_elem_width(ret_vdtype)
        ret = ret.view(ret_vdtype.element_of)
        ret_mask = ret_mask.repeat(2)

    return PyVar(ret, ret_vdtype, ret_mask, r)


@register_ir_api
def vdiv(x, y, mask=None):
    """Computes the division on active elements of ``x`` with the corresponding elements of ``y``.

    - The inactive elements of result vector are undefined.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: 1  4  9  16  25  36  49  64
           y: 1  2  3   4   5   6   7   8
        mask: T  T  F   F   T   T   T   T

         out = S.vdiv(x, y, mask)
         out: 1  2  ?   ?   5   6   7   8

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
        "int8/16/32", "uint8/16/32", "float32".

    Examples
    --------
    .. code-block:: python

        vc = S.vdiv(va, vb)
        vc = S.vdiv(va, 3)
        vc = S.vdiv(va, vb, mask="3T5F")
        vc = S.vdiv(va, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vdiv
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    vdtype = x.dtype
    assert vdtype.is_integer or vdtype.is_float32, "Only support integer and float32 instruction."
    mask = canonicalize_mask(mask, vdtype.lanes)
    return tir.call_extern(vdtype, "__vdiv", PARAM_R_MARK, x, y, mask)


@register_ir_api
def _py_vdiv(x, y, mask=None):
    x, y = broadcast_scalar(x, y)

    ret = x / y
    # For integer, return 0 if divisor = 0;
    # For float, +float/0=inf, -float/0=-inf, 0/0=nan
    if x.dtype.is_integer:
        # Integer vector division will do saturation, because integer division
        # won't underoverflow, so only need care the upper bound.
        ret = np.minimum(ret, get_range(x.dtype)[1])
        ret[y == 0] = 0

    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(ret, x.dtype, mask)


@register_ir_api
def vmod(x, y, mask=None):
    """Computes the remainder on active elements of ``x`` with the corresponding elements of ``y``.

    - The inactive elements of result vector are undefined.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x:  5   6  7  8  -5  -6  -7  -8
           y: -2  -5  4  3   3   4  -3  -4
        mask:  T   T  T  F   F   T   T   T

         out = S.vmod(x, y, mask)
         out:  1   1  3  ?   ?  -2  -1   0

    Parameters
    ----------
    x, y : Union[PrimExpr, int]
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
        "int8/16/32", "uint8/16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vmod(va, vb)
        vc = S.vmod(va, 3)
        vc = S.vmod(va, vb, mask="3T5F")
        vc = S.vmod(va, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vmod
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    vdtype = x.dtype
    assert vdtype.is_integer, "Only support integer instruction."
    mask = canonicalize_mask(mask, vdtype.lanes)
    return tir.call_extern(vdtype, "__vmod", PARAM_R_MARK, x, y, mask)


@register_ir_api
def _py_vmod(x, y, mask=None):
    x, y = broadcast_scalar(x, y)

    ret = np.abs(x.astype("int64")) % np.abs(y.astype("int64"))
    ret = np.where(x < 0, -ret, ret)

    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(ret, x.dtype, mask)


@register_ir_api
def vdot(x, y, mask=None):
    """Computes dot production on every two adjacent elements of ``x`` with the corresponding
    elements of ``y``. The elements of result vector will be saturated.

    - The inactive elements of result vector are undefined.
    - The feature Multiple Width Vector is supported.

    .. code-block::

            x(i16x16): x0  x1  x2  x3  x4  x5  x6  x7       ...  x14  x15
            y(i16x16): y0  y1  y2  y3  y4  y5  y6  y7       ...  y14  y15
        mask(boolx16):  F   F   T   F   F   T   T   T       ...   F    T

           out = S.vdot(x, y, mask)
           out(i32x8):  ?      x2*y2   x5*y5   x6*y6+x7*y7  ...  x15*y15

    Parameters
    ----------
    x, y : Union[PrimExpr, int]
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
    .. code-block::

        # Only supported integer cases:
        case  result dtype  x.dtype   y.dtype
        1     "int16"       "int8"    "int8"
        2     "int16"       "int8"    "uint8"
        3     "int16"       "uint8"   "int8"
        4     "uint16"      "uint8"   "uint8"

        5     "int32"       "int16"   "int16"
        6     "int32"       "int16"   "uint16"
        7     "int32"       "uint16"  "int16"
        8     "uint32"      "uint16"  "uint16"

    Examples
    --------
    .. code-block:: python

        out0 = S.vdot(x, y)
        out1 = S.vdot(x, 3, mask)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vdot
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y, ignore_sign=True)

    x_vdtype, y_vdtype = x.dtype, y.dtype
    assert_not_flexible_width_vector(x_vdtype)
    msg = "Only supports 8-bit or 16-bit integer instruction."
    assert x_vdtype.is_integer and x_vdtype.bits in (8, 16), msg

    ret_vdtype = double_elem_width(x_vdtype)
    ret_vdtype = ret_vdtype.with_int() if x_vdtype.is_int or y_vdtype.is_int else ret_vdtype
    mask = canonicalize_mask(mask, x_vdtype.lanes)
    return tir.call_extern(ret_vdtype, "__vdot", PARAM_R_MARK, x, y, mask)


@register_ir_api
def _py_vdot(x, y, mask=None):
    x, y = broadcast_scalar(x, y)
    x_lanes = x.dtype.lanes
    ret_vdtype = double_elem_width(x.dtype)
    ret_vdtype = ret_vdtype.with_int() if x.dtype.is_int or y.dtype.is_int else ret_vdtype

    mask = canonicalize_mask(mask, x_lanes)
    x, y = x.astype("int64"), y.astype("int64")
    ret = [0] * ret_vdtype.lanes

    for i in range(x_lanes):
        if mask[i]:
            ret[i // 2] += x[i] * y[i]
    ret = np.clip(ret, *get_range(ret_vdtype))

    ret_mask = mask[::2] | mask[1::2]
    return PyVar(ret, ret_vdtype, ret_mask)


@register_ir_api
def vqdot(x, y, mask=None):
    """Computes dot production on every four adjacent elements of ``x`` with the corresponding
    elements of ``y``. The elements of result vector will be saturated.

    - The inactive elements of result vector are undefined.
    - The feature Multiple Width Vector is supported.

    .. code-block::

             x(i8x32): x0  x1  x2  x3  x4  x5  x6  x7           ...  x28  x29  x30  x31
             y(i8x32): y0  y1  y2  y3  y4  y5  y6  y7           ...  y28  y29  y30  y31
        mask(boolx32):  T   F   F   T   T   T   T   T           ...   F    F    F    T

           out = S.vqdot(x, y, mask)
           out(i32x8): x0*y0+x3*y3     x4*y4+x5*y5+x6*y6+x7*y7  ...  x31*y31


           x(fp16x16): x0  x1  x2  x3  x4  x5                  x6  x7  ...  x14  x15
           y(fp16x16): y0  y1  y2  y3  y4  y5                  y6  y7  ...  y14  y15
        mask(boolx16):  F   F   T   F   T   T                   T   T  ...   F    T

          out = vqdot(x, y, mask)
          out(fp32x8): x2*y2    ?      x4*y4+x5*y5+x6*y6+x7*y7  ?      ...   ?
          # For float, the result stores in even index, the values in odd index are undefined.

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
    .. code-block::

        Supported integer cases:
        case  result dtype  x.dtype    y.dtype
        1     "int32"       "int8"     "int8"
        2     "int32"       "int8"     "uint8"
        3     "int32"       "uint8"    "int8"
        4     "uint32"      "uint8"    "uint8"

        Supported floating cases:
        case  result dtype  x.dtype    y.dtype
        1     "float32"     "float16"  "float16"
        2     "float32"     "bfloat16" "bfloat16"

    Examples
    --------
    .. code-block:: python

        out0 = S.vqdot(x, y)
        out1 = S.vqdot(x, 3, mask)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vqdot
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y, ignore_sign=True)

    x_vdtype, y_vdtype = x.dtype, y.dtype
    assert_not_flexible_width_vector(x_vdtype)
    msg = "Only supports 8-bit integer instruction or 16-bit floating instruction."
    assert (x_vdtype.is_integer and x_vdtype.bits == 8) or x_vdtype.is_floating16, msg

    x_lanes = x_vdtype.lanes
    ret_vdtype = double_elem_width(DataType(f"float16x{x_lanes}"))
    if x_vdtype.is_integer:
        ret_vdtype = double_elem_width(double_elem_width(x_vdtype))
        ret_vdtype = ret_vdtype.with_int() if x_vdtype.is_int or y_vdtype.is_int else ret_vdtype
    mask = canonicalize_mask(mask, x_lanes)
    return tir.call_extern(ret_vdtype, "__vqdot", PARAM_R_MARK, x, y, mask)


@register_ir_api
def _py_vqdot(x, y, mask=None):
    x, y = broadcast_scalar(x, y)
    x_lanes = x.dtype.lanes
    ret_vdtype = double_elem_width(DataType(f"float16x{x_lanes}"))
    if x.dtype.is_integer:
        ret_vdtype = double_elem_width(double_elem_width(x.dtype))
        ret_vdtype = ret_vdtype.with_int() if x.dtype.is_int or y.dtype.is_int else ret_vdtype
    mask = canonicalize_mask(mask, x_lanes)
    x = x.astype("int64" if ret_vdtype.is_integer else "float64")
    y = y.astype("int64" if ret_vdtype.is_integer else "float64")
    ret, ret_mask = [0] * ret_vdtype.lanes, [False] * ret_vdtype.lanes

    if ret_vdtype.is_integer:
        for i in range(x_lanes):
            if mask[i]:
                ret[i // 4] += x[i] * y[i]
                ret_mask[i // 4] = True
        ret = np.clip(ret, *get_range(ret_vdtype))
    else:  # float
        for i in range(x_lanes):
            if mask[i]:
                ret[(i // 4) * 2] += x[i] * y[i]
                ret_mask[(i // 4) * 2] = True

    return PyVar(ret, ret_vdtype, ret_mask)


@register_ir_api
def vdpa(acc, x, y, mask=None):
    """Performs an accumulate add operation with every two adjacent elements of inputs.

    - The feature Multiple Width Vector is supported.

    .. code-block::

           acc(i32x8): a0      a1              a2        ...  a7
            x(i16x16): x0  x1  x2  x3          x4  x5    ...  x14  x15
            y(i16x16): y0  y1  y2  y3          y4  y5    ...  y14  y15
        mask(boolx16):  F   F   T   T           T   F    ...   F    T

           out = S.vdpa(acc, x, y, mask)
           out(i32x8): a0      a1+x2*y2+x3*y3  a2+x4*y4  ...  a7+x15*y15

    Parameters
    ----------
    acc : PrimExpr
        The accumulate register, should be initialized.

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
    .. code-block::

        # Only supported integer cases:
        case  acc.dtype  x.dtype   y.dtype
        1     "int16"    "int8"    "int8"
        2     "int16"    "int8"    "uint8"
        3     "int16"    "uint8"   "int8"
        4     "uint16"   "uint8"   "uint8"

        5     "int32"    "int16"   "int16"
        6     "int32"    "int16"   "uint16"
        7     "int32"    "uint16"  "int16"
        8     "uint32"   "uint16"  "uint16"

    Examples
    --------
    .. code-block:: python

        acc = S.int32x8(0)
        out = S.vdpa(acc, x, y)

        acc = S.int32x8(0)
        out = S.vdpa(acc, x, y, mask)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vdpa
    """
    assert is_vector(acc), "The 1st arg expect a vector."
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y, ignore_sign=True)
    acc_vdtype, x_vdtype = acc.dtype, x.dtype
    assert_not_flexible_width_vector(x_vdtype)

    acc_lanes, acc_bits = acc_vdtype.lanes, acc_vdtype.bits
    x_lanes, x_bits = x_vdtype.lanes, x_vdtype.bits
    assert (acc_vdtype.is_integer and x_vdtype.is_integer) and (
        acc_bits * acc_lanes == x_bits * x_lanes
        and acc_bits / x_bits == 2
        and x_lanes / acc_lanes == 2
    ), f'Argument type mismatch: 0-th: "{acc_vdtype}" vs. 1-th: "{x_vdtype}".'

    mask = canonicalize_mask(mask, x_lanes)
    return tir.call_extern(acc_vdtype, "__vdpa", acc, x, y, mask)


@register_ir_api
def _py_vdpa(acc, x, y, mask=None):
    x, y = broadcast_scalar(x, y)
    x_lanes = x.dtype.lanes

    x = x.astype("int64")
    y = y.astype("int64")
    acc_value = acc.astype("int64")
    mask = canonicalize_mask(mask, x_lanes)

    for i in range(x_lanes):
        if mask[i]:
            acc_value[i // 2] += x[i] * y[i]
    acc_value = np.clip(acc_value, *get_range(acc.dtype))
    return PyVar(acc_value, acc.dtype)


@register_ir_api
def vqdpa(acc, x, y, mask=None):
    """Performs an accumulate add operation with every four adjacent elements of inputs.

    - The feature Multiple Width Vector is supported.

    .. code-block::

           acc(i32x8): a0              a1              a2                ...   a7
             x(i8x32): x0  x1  x2  x3  x4  x5  x6  x7  x8  x9  x10  x11  ...  x28  x29  x30  x31
             y(i8x32): y0  y1  y2  y3  y4  y5  y6  y7  y8  y9  y10  y11  ...  y28  y29  y30  y31
        mask(boolx32):  T   F   F   F   T   F   T   F   F   T   F    F   ...   T    F    F    T

           out = S.vqdpa(acc, x, y, mask)
           out(i32x8): a0+x0*y0        a1+x4*y4+x6*y6  a2+x9*y9          ...  a7+x28*y28+x31*y31


          acc(fp32x8): a0              a1       a2                          a3      ...   a7
           x(fp16x16): x0  x1          x2   x3  x4  x5                      x6  x7  ...  x14  x15
           y(fp16x16): y0  y1          y2   y3  y4  y5                      y6  y7  ...  y14  y15
        mask(boolx16):  T   F           F    T   T   T                       T   T  ...   F    T

          out = S.vqdpa(acc, x, y, mask)
          out(fp32x8): a0+x0*y0+x3*y3  a1       a2+x4*y4+x5*y5+x6*y6+x7*y7  a3      ...   a7
          # For float, the result stores in even index, odd index keep the value of "acc".

    Parameters
    ----------
    acc : PrimExpr
        The accumulate register, should be initialized.

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
    .. code-block::

        Supported integer cases:
        case  acc.dtype  x.dtype    y.dtype
        1     "int32"    "int8"     "int8"
        2     "int32"    "int8"     "uint8"
        3     "int32"    "uint8"    "int8"
        4     "uint32"   "uint8"    "uint8"

        Supported floating cases:
        case  acc.dtype  x.dtype    y.dtype
        1     "float32"  "float16"  "float16"
        2     "float32"  "bfloat16"  "bfloat16"

    Examples
    --------
    .. code-block:: python

        acc = S.int32x8(0)
        out = S.vqdpa(acc, x, y)

        acc = S.int32x8(0)
        out = S.vqdpa(acc, x, y, mask)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vqdpa
    """
    assert is_vector(acc), "The 1st arg expect a vector."
    cps_info = CompassInfo.current()
    if acc.dtype.is_float:
        msg = f'The "vqdpa" does not support the target "{cps_info.name}".'
        assert cps_info.version == "X2", msg

    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y, ignore_sign=True)
    acc_vdtype, x_vdtype = acc.dtype, x.dtype
    assert_not_flexible_width_vector(x_vdtype)

    acc_lanes, acc_bits = acc_vdtype.lanes, acc_vdtype.bits
    x_lanes, x_bits = x_vdtype.lanes, x_vdtype.bits

    def is_valid_lanes_bits(count):
        return (
            acc_bits * acc_lanes == x_bits * x_lanes
            and acc_bits / x_bits == count
            and x_lanes / acc_lanes == count
        )

    assert ((acc_vdtype.is_integer and x_vdtype.is_integer) and is_valid_lanes_bits(4)) or (
        (acc_vdtype.is_float and x_vdtype.is_floating) and is_valid_lanes_bits(2)
    ), f'Argument type mismatch: 0-th: "{acc_vdtype}" vs. 1-th: "{x_vdtype}".'

    mask = canonicalize_mask(mask, x_lanes)
    return tir.call_extern(acc_vdtype, "__vqdpa", acc, x, y, mask)


@register_ir_api
def _py_vqdpa(acc, x, y, mask=None):
    x, y = broadcast_scalar(x, y)
    x_lanes = x.dtype.lanes

    mask = canonicalize_mask(mask, x_lanes)

    if acc.dtype.is_integer:
        x = x.astype("int64")
        y = y.astype("int64")
        acc_value = acc.astype("int64")
        for i in range(x_lanes):
            if mask[i]:
                acc_value[i // 4] += x[i] * y[i]
        acc_value = np.clip(acc_value, *get_range(acc.dtype))
    else:  # float
        x = x.astype("float64")
        y = y.astype("float64")
        acc_value = acc.astype("float64")
        for i in range(x_lanes):
            if mask[i]:
                acc_value[(i // 4) * 2] += x[i] * y[i]
    return PyVar(acc_value, acc.dtype)


@register_ir_api
def vrpadd(x, mask=None):
    """Computes the reduction addition of all active elements of ``x``, and places the result as the
    lowest elements of result vector.

    - The feature Multiple Width Vector is supported.
    - The remaining upper elements of result vector are undefined.

    .. code-block::

           x:  0  1  2  3  4  5  6  7
        mask:  T  T  T  T  T  T  F  T

         out = S.vrpadd(x, mask)
         out: 22  ?  ?  ?  ?  ?  ?  ?

    Parameters
    ----------
    x : PrimExpr,
        The operands.

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

        out = S.vrpadd(x)
        out = S.vrpadd(x, mask)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vrpadd
    """
    vdtype = x.dtype
    assert_not_flexible_width_vector(vdtype)
    ret_vdtype = f"float32x{vdtype.lanes//2}" if vdtype.is_floating16 else vdtype
    mask = canonicalize_mask(mask, vdtype.lanes)
    return tir.call_extern(ret_vdtype, "__vrpadd", PARAM_R_MARK, x, mask)


@register_ir_api
def _py_vrpadd(x, mask=None):
    mask = canonicalize_mask(mask, x.dtype.lanes)
    ret_dtype = DataType("float32") if x.dtype.is_floating16 else x.dtype.element_of
    ret_lanes = (x.dtype.lanes // 2) if x.dtype.is_floating16 else x.dtype.lanes
    ret = PyVar.zeros(ret_dtype.with_lanes(ret_lanes))

    # Use numpy sum for 2 reasons:
    # 1. Same behavior as hardware.
    # 2. Pairwise summation gets higher precision.
    hw_lanes = hw_native_vdtype(x.dtype).lanes
    zero = np.array(0, x.dtype.element_of)
    for i in range(x.dtype.lanes // hw_lanes):
        start = i * hw_lanes
        mask_i, x_i = mask[start : start + hw_lanes], x[start : start + hw_lanes]
        ret[0] += np.sum(np.where(mask_i, x_i, zero), dtype=ret_dtype)
    return ret


@register_ir_api
def vmml(ptr, x, y):
    """Performs a mixture precision 4x4 matrix multiply and addition for float16x16/bfloat16x16
    vector x (row-major) and y (column-major). The result pointer ``ptr`` is the address of
    float32x16 with row-major. This behavior is the same as ``ptr[:?] = matrix_multiply(x, y)``.

    .. code-block::

                              y(fp16x16):  y0   y4   y8  y12
                                           y1   y5   y9  y13
                                           y2   y6  y10  y14
                                           y3   y7  y11  y15

        x(fp16x16):  x0   x1   x2   x3     a0   a1   a2   a3 :ptr(fp32x16)
                     x4   x5   x6   x7     a4   a5   a6   a7
                     x8   x9  x10  x11     a8   a9  a10  a11
                    x12  x13  x14  x15    a12  a13  a14  a15

        S.vmml(ptr, x, y)

        # Detailed computation for each result element:
        a0 = x0*y0 + x1*y1 + x2*y2 + x3*y3
        a1 = x0*y4 + x1*y5 + x2*y6 + x3*y7
        ...
        a9 = x8*y4 + x9*y5 + x10*y6 + x11*y7
        ...
        a15 = x12*y12 + x13*y13 + x14*y14 + x15*y15

    Parameters
    ----------
    ptr : Pointer
        The pointer that store the memory address in where the result will be stored, it can be a
        scalar or vector float32 pointer, the memory space it point to at least must can represent a
        4x4 float32 matrix with row major.

    x : PrimExpr
        The operand x with vector type float16x16/bfloat16x16 representing 4x4 fp16/bf16 elements
        with row major.

    y : PrimExpr
        The operand y with vector type float16x16/bfloat16x16 representing 4x4 fp16/bf16 elements
        with column major.

    Supported DType
    ---------------
        "float16", "bfloat16".

    Examples
    --------
    .. code-block:: python

        # The "vc_fp32_ptr" can be scalar or vector float32 pointer, as long as the memory space
        # that it point to is enough to store 4x4 float32 data.
        S.vmml(vc_fp32_ptr, va_fp16x16, vb_fp16x16)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vmml
    """
    cps_info = CompassInfo.current()
    msg = f'The "vmml" does not support the target "{cps_info.name}".'
    assert cps_info.version == "X2", msg

    assert isinstance(ptr, (tir.Buffer, tir.Pointer)), "The 1st arg expect a pointer."
    ptr = ptr.addr_of(0) if isinstance(ptr, tir.Buffer) else ptr
    msg = f'The 1st arg expect a float32 pointer, but got: "{ptr.dtype}*".'
    assert ptr.dtype.is_float32, msg
    assert is_vector(x) and is_vector(y), "The 2nd and 3rd arg all expect a vector."
    # Shouldn't support broadcast scalar here, because for matrix multiply, all inputs should be
    # explicitly defined as matrix.
    msg = f'The data type of the 2nd arg expect "float16x16/bfloat16x16", but got: "{x.dtype}".'
    assert x.dtype.endswith("float16x16"), msg
    msg = f'The data type of the 3rd arg expect "float16x16/bfloat16x16", but got: "{y.dtype}".'
    assert y.dtype.endswith("float16x16"), msg
    return tir.call_extern("void", "__vmml", ptr, x, y)


@register_ir_api
def _py_vmml(ptr, x, y):
    inputs = (("in0", x.value), ("in1", y.value))
    outputs = (("out", ptr),)
    vdtype = "__bf1616" if x.dtype == "bfloat16x16" else "half16"
    code_snippet = f"__vmml(out, ((__global {vdtype}*)in0)[0], ((__global {vdtype}*)in1)[0]);"
    pysim_run_sim(code_snippet, inputs, outputs)


@register_ir_api
def vmma(acc_ptr, x, y):
    """Performs a mixture precision 4x4 matrix multiply and addition for float16x16/bfloat16x16
    vector x (row-major) and y (column-major). The result pointer ``acc_ptr`` is the address of
    float32x16 with row-major.
    This behavior is the same as``acc_ptr[:?] += matrix_multiply(x, y)``.

    .. code-block::

                              y(fp16x16):  y0   y4   y8  y12
                                           y1   y5   y9  y13
                                           y2   y6  y10  y14
                                           y3   y7  y11  y15

        x(fp16x16):  x0   x1   x2   x3     a0   a1   a2   a3 :acc_ptr(fp32x16)
                     x4   x5   x6   x7     a4   a5   a6   a7
                     x8   x9  x10  x11     a8   a9  a10  a11
                    x12  x13  x14  x15    a12  a13  a14  a15

        S.vmma(acc_ptr, x, y)

        # Detailed computation for each result element:
        a0 += x0*y0 + x1*y1 + x2*y2 + x3*y3
        a1 += x0*y4 + x1*y5 + x2*y6 + x3*y7
        ...
        a9 += x8*y4 + x9*y5 + x10*y6 + x11*y7
        ...
        a15 += x12*y12 + x13*y13 + x14*y14 + x15*y15

    Parameters
    ----------
    acc_ptr : Pointer
        The pointer that store the memory address in where the result will be stored, it can be a
        scalar or vector float32 pointer, the memory space it point to at least must can represent a
        4x4 float32 matrix with row major.

    x : PrimExpr
        The operand x with vector type float16x16/bfloat16x16 representing 4x4 fp16/bf16 elements
        with row major.

    y : PrimExpr
        The operand y with vector type float16x16/bfloat16x16 representing 4x4 fp16/bf16 elements
        with column major.

    Supported DType
    ---------------
        "float16", "bfloat16".

    Examples
    --------
    .. code-block:: python

        # The "vc_fp32_ptr" can be scalar or vector float32 pointer, as long as the memory space
        # that it point to is enough to store 4x4 float32 data.
        S.vmma(vc_fp32_ptr, va_fp16x16, vb_fp16x16)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vmma
    """
    cps_info = CompassInfo.current()
    msg = f'The "vmma" does not support the target "{cps_info.name}".'
    assert cps_info.version == "X2", msg

    assert isinstance(acc_ptr, (tir.Buffer, tir.Pointer)), "The 1st arg expect a pointer."
    acc_ptr = acc_ptr.addr_of(0) if isinstance(acc_ptr, tir.Buffer) else acc_ptr
    msg = f'The 1st arg expect a float32 pointer, but got: "{acc_ptr.dtype}*".'
    assert acc_ptr.dtype.is_float32, msg
    assert is_vector(x) and is_vector(y), "The 2nd and 3rd arg all expect a vector."
    # Shouldn't support broadcast scalar here, because for matrix multiply, all inputs should be
    # explicitly defined as matrix.
    msg = f'The data type of the 2nd arg expect "float16x16/bfloat16x16", but got: "{x.dtype}".'
    assert x.dtype.endswith("float16x16"), msg
    msg = f'The data type of the 3rd arg expect "float16x16/bfloat16x16", but got: "{y.dtype}".'
    assert y.dtype.endswith("float16x16"), msg
    return tir.call_extern("void", "__vmma", acc_ptr, x, y)


@register_ir_api
def _py_vmma(acc_ptr, x, y):
    inputs = (("in0", x.value), ("in1", y.value))
    outputs = (("out", acc_ptr),)
    vdtype = "__bf1616" if x.dtype == "bfloat16x16" else "half16"
    code_snippet = f"__vmma(out, ((__global {vdtype}*)in0)[0], ((__global {vdtype}*)in1)[0]);"
    pysim_run_sim(code_snippet, inputs, outputs)


def _vfma(acc, x, y, mask=None):
    assert is_vector(acc), "The 1st arg expect a vector in vector situation."
    acc, x, y = broadcast_scalar(acc, x, y)
    assert_vdtype_match(acc, x, y)
    acc_vdtype = acc.dtype
    assert acc_vdtype.is_float32, "Only support float32 instruction."
    mask = canonicalize_mask(mask, acc_vdtype.lanes)
    return tir.call_extern(acc_vdtype, "__vfma", acc, x, y, mask)


def _py_vfma(acc, x, y, mask=None):
    acc, x, y = broadcast_scalar(acc, x, y)
    mask = canonicalize_mask(mask, acc.dtype.lanes)

    acc_fp64, x, y = acc.astype("float64"), x.astype("float64"), y.astype("float64")
    acc[mask] = (acc_fp64 + x * y).astype("float32")[mask]
    return acc


@register_ir_api
def fma(acc, x, y, mask=None):
    """Performs float multiply-add operation on every active elements of inputs.

    - The scalar situation where all of ``acc``, ``x`` and ``y`` are scalar is also supported.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

         acc: a0     a1        a2     ...     a7
           x: x0     x1        x2     ...     x7
           y: y0     y1        y2     ...     y7
        mask:  F      T         T     ...      T

         out = S.fma(acc, x, y, mask)
         out: a0  a0+x1*y1  a2+x2*y2  ...  a7+x7*y7

    Parameters
    ----------
    acc : PrimExpr
        The accumulate register, should be initialized.

    x, y : Union[PrimExpr, float]
        The operands. If it is a scalar in the vector situation, it will be automatically
        broadcast.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "float32".

    Examples
    --------
    .. code-block:: python

        acc = S.float32x8(10)
        out = S.fma(acc, x, y, mask)

        scalar_out = S.fma(scalar_acc, scalar_x, scalar_y)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vfma, __fma
    """
    if all(is_scalar(arg) for arg in (acc, x, y)):
        x, y = binary_op_match_types(x, y)
        acc, x = binary_op_match_types(acc, x)

        acc_dtype = acc.dtype
        msg = "Only support float32 instruction."
        assert acc_dtype.is_float32 and y.dtype.is_float32, msg

        return tir.call_extern(acc_dtype, "__fma", acc, x, y)
    return _vfma(acc, x, y, mask)


@register_ir_api
def _py_fma(acc, x, y, mask=None):
    if is_scalar(acc):
        acc_fp64, x, y = (np.array(arg, "float64") for arg in (acc, x, y))
        return PyVar(acc_fp64 + x * y, dtype=DataType("float32"))
    return _py_vfma(acc, x, y, mask)


@register_ir_api
def vfmae(acc, x, y, mask=None):
    """Performs float multiply-add operation on even active elements of inputs.

    .. code-block::

         acc(fp32x8): a0      a1        a2        ...   a7
          x(fp16x16): x0  x1  x2  x3    x4  x5    ...  x14  x15
          y(fp16x16): y0  y1  y2  y3    y4  y5    ...  y14  y15
        mask(boolx8):  F       T         T        ...   T

         out = S.vfmae(acc, x, y, mask)
         out(fp32x8): a0      a1+x2*y2  a2+x4*y4  ...  a7+x14*y14

    Parameters
    ----------
    acc : PrimExpr
        The accumulate register, should be initialized.

    x, y : Union[PrimExpr, float]
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
    .. code-block::

        # Only supported floating cases:
        case  acc.dtype  x.dtype    y.dtype
        1     "float32"  "float16"  "float16"
        2     "float32"  "bfloat16"  "bfloat16"

    Examples
    --------
    .. code-block:: python

        acc = S.float32x8(10)
        out = S.vfmae(acc, x, y)

        acc = S.float32x8(10)
        out = S.vfmae(acc, x, y, mask)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vfmae
    """
    assert is_vector(acc), "The 1st arg expect a vector."
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)

    acc_vdtype, x_vdtype = acc.dtype, x.dtype
    assert_neither_flexible_nor_multiple_width_vector(x_vdtype)
    assert (acc_vdtype.is_float32 and x_vdtype.is_floating16) and (
        x_vdtype.lanes / acc_vdtype.lanes == 2
    ), f'Argument type mismatch: 0-th: "{acc_vdtype}" vs. 1-th: "{x_vdtype}".'

    mask = canonicalize_mask(mask, acc_vdtype.lanes)
    return tir.call_extern(acc_vdtype, "__vfmae", acc, x, y, mask)


@register_ir_api
def _py_vfmae(acc, x, y, mask=None):
    x, y = broadcast_scalar(x, y)

    x = x.astype(acc.dtype.element_of)
    y = y.astype(acc.dtype.element_of)

    mask = canonicalize_mask(mask, acc.dtype.lanes)

    for acc_idx in range(acc.dtype.lanes):
        xy_idx = acc_idx * 2
        if mask[acc_idx]:
            acc[acc_idx] += x[xy_idx] * y[xy_idx]
    return acc


@register_ir_api
def vfmao(acc, x, y, mask=None):
    """Performs float multiply-add operation on odd active elements of inputs.

    .. code-block::

         acc(fp32x8): a0      a1        a2        ...   a7
          x(fp16x16): x0  x1  x2  x3    x4  x5    ...  x14  x15
          y(fp16x16): y0  y1  y2  y3    y4  y5    ...  y14  y15
        mask(boolx8):  F       T         T        ...   T

         out = S.vfmao(acc, x, y, mask)
         out(fp32x8): a0      a1+x3*y3  a2+x5*y5  ...  a7+x15*y15

    Parameters
    ----------
    acc : PrimExpr
        The accumulate register, should be initialized.

    x, y : Union[PrimExpr, float]
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
    .. code-block::

        # Only supported floating cases:
        case  acc.dtype  x.dtype    y.dtype
        1     "float32"  "float16"  "float16"
        2     "float32"  "bfloat16"  "bfloat16"

    Examples
    --------
    .. code-block:: python

        acc = S.float32x8(10)
        out = S.vfmao(acc, x, y)

        acc = S.float32x8(10)
        out = S.vfmao(acc, x, y, mask)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vfmao
    """
    assert is_vector(acc), "The 1st arg expect a vector."
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)

    acc_vdtype, x_vdtype = acc.dtype, x.dtype
    assert_neither_flexible_nor_multiple_width_vector(x_vdtype)
    assert (acc_vdtype.is_float32 and x_vdtype.is_floating16) and (
        x_vdtype.lanes / acc_vdtype.lanes == 2
    ), f'Argument type mismatch: 0-th: "{acc_vdtype}" vs. 1-th: "{x_vdtype}".'

    mask = canonicalize_mask(mask, acc_vdtype.lanes)
    return tir.call_extern(acc_vdtype, "__vfmao", acc, x, y, mask)


@register_ir_api
def _py_vfmao(acc, x, y, mask=None):
    x, y = broadcast_scalar(x, y)

    x = x.astype(acc.dtype.element_of)
    y = y.astype(acc.dtype.element_of)

    mask = canonicalize_mask(mask, acc.dtype.lanes)

    for acc_idx in range(acc.dtype.lanes):
        xy_idx = acc_idx * 2 + 1
        if mask[acc_idx]:
            acc[acc_idx] += x[xy_idx] * y[xy_idx]
    return acc


@register_ir_api
def rint(x, mask=None):
    """Computes the rounding on active elements of ``x``.

    - The scalar situation where ``x`` is a scalar is also supported.
    - The inactive elements of result vector are undefined.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: -0.4  0.2  1.4  1.5  1.6  1.8  1.9  2.01
        mask:   T    T    T    T    T    F    T    T

         out = S.rint(x, mask)
         out: -0.0  0.0  1.0  2.0  2.0   ?   2.0  2.0

    Parameters
    ----------
    x : Union[PrimExpr, float]
        The operands. The vector x.

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

        vc = S.rint(va)
        vc = S.rint(va, mask="3T5F")
        vc = S.rint(va, mask=S.tail_mask(n, 8))
        scalar_c = S.rint(1.23456)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vrint, __rint
    """
    x_dtype = get_dtype(x)
    assert x_dtype.is_floating, "Only support floating input."
    if x_dtype.is_scalar:
        assert_not_bfloat16_scalar(x_dtype)
        assert mask is None, "Only support mask in vector scenario."
        return tir.call_extern(x_dtype, "__rint", x)

    # Vector scenario.
    mask = canonicalize_mask(mask, x_dtype.lanes)
    return tir.call_extern(x_dtype, "__vrint", PARAM_R_MARK, x, mask)


@register_ir_api
def _py_rint(x, mask=None):
    x_dtype = get_dtype(x)
    mask = canonicalize_mask(mask, x_dtype.lanes)
    return PyVar(np.rint(x), x_dtype, mask)


def _try_shrink_to_meaningful_range(min_val, max_val, dtype):
    meaningful_min, meaningful_max = get_range(dtype)
    min_val = max(meaningful_min, min_val) if is_scalar_const(min_val) else min_val
    max_val = min(meaningful_max, max_val) if is_scalar_const(max_val) else max_val
    return min_val, max_val


@register_ir_api
def clip(x, min_val, max_val, mask=None):
    """Clip active elements of ``x`` with the corresponding elements of ``min_val`` and
    ``max_val``.

    - The scalar situation where all of ``x``, ``min_val``, ``max_val`` are scalar is also
      supported.
    - The inactive elements of result vector are set to the corresponding elements of ``x``.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

              x: 1  3  4  9  4  4  8  8
        min_val: 3  3  3  3  5  5  5  5
        max_val: 8  8  8  8  7  7  7  7
           mask: T  T  T  T  F  T  F  T

        out = S.clip(x, min_val, max_val, mask)
            out: 3  3  4  8  4  5  8  7

    Parameters
    ----------
    x, min_val, max_val : Union[PrimExpr, int, float]
        The operands. If either one is a scalar, it will be automatically broadcast. It should be
        noted that: min_val < max_val.

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

        b = S.clip(a, -10, 10)
        vc = S.clip(va, vb, vc)
        vc = S.clip(va, 3, 30)
        vc = S.clip(va, vb, vc, mask="3T5F")
        vc = S.clip(va, vb, vc, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vmax, __vmin, __vsel, __vclt, __vcgt
    """
    from .conversion import cast  # pylint: disable=import-outside-toplevel
    from .compare import max, min  # pylint: disable=import-outside-toplevel,redefined-builtin
    from .compare import vclt, vcgt  # pylint: disable=import-outside-toplevel
    from .permutation import vsel  # pylint: disable=import-outside-toplevel

    is_all_scalar = all(is_scalar(arg) for arg in (x, min_val, max_val))
    if is_all_scalar:
        assert isinstance(x, tir.PrimExpr), "The 1st arg expect a variable."
        assert_not_bfloat16_scalar(x.dtype)
    else:
        assert is_vector(x), "The 1st arg expect a vector."

    x_dtype = x.dtype
    min_val, max_val = _try_shrink_to_meaningful_range(min_val, max_val, x_dtype)
    assert can_safe_cast(min_val, x_dtype), f'Can\'t cast the arg "min_val" to "{x_dtype}" safely.'
    assert can_safe_cast(max_val, x_dtype), f'Can\'t cast the arg "max_val" to "{x_dtype}" safely.'
    min_val, max_val = cast(min_val, x_dtype), cast(max_val, x_dtype)

    if is_all_scalar:
        return max(min(x, max_val), min_val)

    # Vector scenario.
    if x_dtype.is_integer:
        return max(min(x, max_val, mask, r=x), min_val, mask, r=x)

    # For float, using "vsel" instead of "vmax" and "vmin", because the Zhouyi NPU behavior of
    # "vmax" or "vmin" is unexpected when one operand is NaN, and another is a normal value, its
    # result is the normal value. In other words, "vmax" or "vmin" doesn't reserve NaN.
    vout_ge_min = vsel(min_val, x, vclt(x, min_val, mask))
    return vsel(max_val, vout_ge_min, vcgt(x, max_val, mask))


@register_ir_api
def _py_clip(x, min_val, max_val, mask=None):
    if all(is_scalar(arg) for arg in (x, min_val, max_val)):
        return PyVar(np.clip(x, min_val, max_val), x.dtype)

    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(np.clip(x, min_val, max_val), x.dtype, mask, x)


__all__ = (
    "vadd",
    "vaddh",
    "abs",
    "vsub",
    "vsubh",
    "vmul",
    "vmull",
    "vmulh",
    "vdiv",
    "vmod",
    "vdot",
    "vqdot",
    "vdpa",
    "vqdpa",
    "vrpadd",
    "vmml",
    "vmma",
    "fma",
    "vfmae",
    "vfmao",
    "rint",
    "clip",
)
