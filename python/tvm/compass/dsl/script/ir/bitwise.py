# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The bitwise part of IR APIs."""
import numpy as np
from tvm import tir, get_range
from ..pysim import PyVar
from .base import register_ir_api, canonicalize_mask
from .utils import PARAM_R_MARK, broadcast_scalar, assert_vdtype_match, change_sign_if_needed
from .utils import is_vector, is_vector_or_mask, get_dtype, canonicalize_r
from .utils import assert_not_flexible_width_vector


@register_ir_api
def vand(x, y, mask=None):
    """Computes the bit-wise AND on active elements of ``x`` with the corresponding elements of
    ``y``.

    - The mask situation where both ``x`` and ``y`` are mask is also supported.
    - The inactive elements of result vector are set to False for mask situation, undefined for
      others.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x:     1         2         3         4         5        6         7         8
              00000001  00000010  00000011  00000100  00000101  00000110  00000111  00001000
           y:     5         6         7         8        -9       -10       -11       -12
              00000101  00000110  00000111  00001000  11110111  11110110  11110101  11110100
        mask:     T         T         F         T         T        F         T         T

         out = S.vand(x, y, mask)
         out:     1         2         ?         0         5        ?         5         0
              00000001  00000010      ?     00000000  00000101     ?      00000101  00000000

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
        "int8/16/32", "uint8/16/32", "bool8/16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vand(va, vb)
        vc = S.vand(va, 3)
        vc = S.vand(va, vb, mask="3T5F")
        vc = S.vand(va, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vand
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    vdtype = x.dtype
    assert vdtype.is_integer, "Only support integer instruction."
    mask = canonicalize_mask(mask, vdtype.lanes)
    return tir.call_extern(vdtype, "vand", PARAM_R_MARK, x, y, mask)


@register_ir_api
def _py_vand(x, y, mask=None):
    x, y = broadcast_scalar(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(x & y, x.dtype, mask, False if x.dtype.is_bool else None)


@register_ir_api
def vor(x, y, mask=None):
    """Computes the bit-wise OR on active elements of ``x`` with the corresponding elements of
    ``y``.

    - The mask situation where both ``x`` and ``y`` are mask is also supported.
    - The inactive elements of result vector are set to False for mask situation, undefined for
      others.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x:     1         2         3         4         5        6         7         8
              00000001  00000010  00000011  00000100  00000101  00000110  00000111  00001000
           y:     5         6         7         8        -9       -10       -11       -12
              00000101  00000110  00000111  00001000  11110111  11110110  11110101  11110100
        mask:     T         T         F         T         T        F         T         T

         out = S.vor(x, y, mask)
         out:     5         6         ?        12        -9        ?        -9        -4
              00000101  00000110      ?     00001100  11110111     ?      11110111  11111100

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
        "int8/16/32", "uint8/16/32", "bool8/16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vor(va, vb)
        vc = S.vor(va, 3)
        vc = S.vor(va, vb, mask="3T5F")
        vc = S.vor(va, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vor
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    vdtype = x.dtype
    assert vdtype.is_integer, "Only support integer instruction."
    mask = canonicalize_mask(mask, vdtype.lanes)
    return tir.call_extern(vdtype, "vor", PARAM_R_MARK, x, y, mask)


@register_ir_api
def _py_vor(x, y, mask=None):
    x, y = broadcast_scalar(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(x | y, x.dtype, mask, False if x.dtype.is_bool else None)


@register_ir_api
def vinv(x, mask=None):
    """Computes the bit-wise inversion or bit-wise NOT on active elements of ``x``.

    - The mask situation where ``x`` is a mask is also supported.
    - The inactive elements of result vector are set to False for mask situation, undefined for
      others.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x:     1         2         3         4         5         6         7         8
              00000001  00000010  00000011  00000100  00000101  00000110  00000111  00001000
        mask:     T         F         F         T         T         T         T         T

         out = S.vinv(x, mask)
         out:    -2         ?         ?        -5        -6        -7        -8        -9
              11111110      ?         ?     11111011  11111010  11111001  11111000  11110111

    Parameters
    ----------
    x : PrimExpr
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
        "int8/16/32", "uint8/16/32", "bool8/16/32".

    Examples
    --------
    .. code-block:: python

        vr = S.vinv(vx)
        vr = S.vinv(vx, mask="3T5F")
        vr = S.vinv(vx, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vinv
    """
    assert is_vector_or_mask(x), "The 1st arg expect a vector or a mask."
    vdtype = x.dtype
    assert vdtype.is_integer, "Only support integer instruction."
    mask = canonicalize_mask(mask, vdtype.lanes)
    return tir.call_extern(vdtype, "vinv", PARAM_R_MARK, x, mask)


@register_ir_api
def _py_vinv(x, mask=None):
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(~x, x.dtype, mask, False if x.dtype.is_bool else None)


@register_ir_api
def vall(mask):
    """Computes the logical AND on each element of ``mask``, test whether all elements are True.

    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

        mask(boolx8): T  T  T  T  F  F  F  F

           out = S.vall(mask)
           out(bool): F

    Parameters
    ----------
    mask : PrimExpr
        The operand.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "bool8/16/32".

    Examples
    --------
    .. code-block:: python

        out = S.vall(mask_var)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vandr
    """
    assert isinstance(mask, tir.PrimExpr), 'The arg "mask" expect a variable.'
    vdtype = mask.dtype
    assert vdtype.is_bool_vector, 'The arg "mask" expect a boolean vector, but got: "{vdtype}".'
    return tir.call_extern("bool", "vall", mask)


@register_ir_api
def _py_vall(mask):
    return bool(np.all(mask))


@register_ir_api
def vany(mask):
    """Computes the logical OR on each element of ``mask``, test whether any element is True.

    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

        mask(boolx8): T  T  T  T  F  F  F  F

           out = S.vany(mask)
           out(bool): T

    Parameters
    ----------
    mask : PrimExpr
        The operand.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "bool8/16/32".

    Examples
    --------
    .. code-block:: python

        out = S.vany(mask_var)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vorr
    """
    assert isinstance(mask, tir.PrimExpr), 'The arg "mask" expect a variable.'
    vdtype = mask.dtype
    assert vdtype.is_bool_vector, 'The arg "mask" expect a boolean vector, but got: "{vdtype}".'
    return tir.call_extern("bool", "vany", mask)


@register_ir_api
def _py_vany(mask):
    return bool(np.any(mask))


@register_ir_api
def vxor(x, y, mask=None):
    """Computes the bit-wise XOR on active elements of ``x`` with the corresponding elements of
    ``y``.

    - The mask situation where both ``x`` and ``y`` are mask is also supported.
    - The inactive elements of result vector are set to False for mask situation, undefined for
      others.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x:     1         2         3         4        5         6         7         8
              00000001  00000010  00000011  00000100  00000101  00000110  00000111  00001000
           y:     5         6         7         8       -9        -10       -11       -12
              00000101  00000110  00000111  00001000  11110111  11110110  11110101  11110100
        mask:     T         T         F         T        T         T         T         T

         out = S.vxor(x, y, mask)
         out:     4         4         ?        12       -14       -16       -14       -4
              00000100  00000100      ?     00001100  11110010  11110000  11110010  11111100

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
        "int8/16/32", "uint8/16/32", "bool8/16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vxor(va, vb)
        vc = S.vxor(va, 3)
        vc = S.vxor(va, vb, mask="3T5F")
        vc = S.vxor(va, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vxor
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    vdtype = x.dtype
    assert vdtype.is_integer, "Only support integer instruction."
    mask = canonicalize_mask(mask, vdtype.lanes)
    return tir.call_extern(vdtype, "vxor", PARAM_R_MARK, x, y, mask)


@register_ir_api
def _py_vxor(x, y, mask=None):
    x, y = broadcast_scalar(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(x ^ y, x.dtype, mask, False if x.dtype.is_bool else None)


@register_ir_api
def vnsr(x, shift, mask=None, saturate=False, out_sign=None, with_round=False, to_h=False):
    """Performs a bit-wise right shift on active elements of ``x``, using the corresponding elements
    of ``shift`` as the shift value, then cast the result to the needed narrower bits data type.

    Please note that after casting, the meaningful elements of result vector are still in the
    original position, i.e., they are stored in intervals, not compactly, the meaningless elements
    as intervals are set to zero.

    - The inactive elements of result vector are set to zero.
    - The feature Multiple Width Vector is supported.

    .. code-block::

            x(i32x8):    -9           46            7        ...      6           95
                      11110111     00101110     00000111     ...  00000110     01011111
        shift(i32x8):     2            1            1        ...      2           -2
        mask(boolx8):     T            T            T        ...      F            T

         out = S.vnsr(x, shift, mask, saturate=True, out_sign="u", with_round=True, to_h=True)
         out(u16x16):     0     0     23     0      4     0  ...      0     0      0     0
                      00000000     00010111     00000100          00000000     00000000

    Parameters
    ----------
    x : PrimExpr
        The operand.

    shift : Union[PrimExpr, int]
        The shift value. If it is a scalar, it will be automatically broadcast. The negative number
        will be reinterpreted as an unsigned number.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    saturate : Optional[bool]
        Whether the result needs to be saturated or not.

    out_sign : Optional[str]
        Specify whether the output sign is signed or unsigned. ``None`` means same as ``x``, ``u``
        means unsigned, ``s`` means signed.

    with_round : Optional[bool]
        Whether the result needs to be rounded or not.

    to_h : Optional[bool]
        Whether the output type is short.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int16/32", "uint16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vnsr(va, vb)
        vc = S.vnsr(va, 3)
        vc = S.vnsr(va, vb, mask="3T5F")
        vc = S.vnsr(va, vb, mask=S.tail_mask(n, 8))
        vc = S.vnsr(va, vb, saturate=True)
        vc = S.vnsr(va, vb, out_sigh='s')
        vc = S.vnsr(va, vb, with_round=True)
        vc = S.vnsr(va, vb, to_h=True)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vnsrl, __vnsrlr, __vnsrls, __vnsrlsr, __vnsra,
      __vnsrar, __vnsras, __vnsrasr
    """
    x, shift = broadcast_scalar(x, shift)
    assert_vdtype_match(x, shift, ignore_sign=True)
    x_vdtype = x.dtype
    assert_not_flexible_width_vector(x_vdtype)
    msg = "Only support 16/32 bits integer instruction."
    assert x_vdtype.is_integer and x_vdtype.bits in (16, 32), msg

    ret_vdtype = x_vdtype.with_bits(16 if x_vdtype.bits == 32 and to_h else 8)
    ret_vdtype = ret_vdtype.with_lanes(x_vdtype.total_bits // ret_vdtype.bits)
    ret_vdtype = change_sign_if_needed(ret_vdtype, out_sign)
    if to_h:
        assert x_vdtype.bits == 32, 'The arg "to_h" only can be set for 32 bits instruction.'

    mask = canonicalize_mask(mask, x_vdtype.lanes)
    return tir.call_extern(ret_vdtype, "vnsr", x, shift, mask, saturate, with_round)


@register_ir_api
def _py_vnsr(x, shift, mask=None, saturate=False, out_sign=None, with_round=False, to_h=False):
    x, shift = broadcast_scalar(x, shift)
    ret_vdtype = x.dtype.with_bits(16 if x.dtype.bits == 32 and to_h else 8)
    ret_vdtype = ret_vdtype.with_lanes(x.dtype.total_bits // ret_vdtype.bits)
    ret_vdtype = change_sign_if_needed(ret_vdtype, out_sign)

    if with_round:
        shift_out = np.around(x * (0.5 ** shift.astype("uint32"))).astype("int64")
    else:
        shift_out = x >> shift

    if saturate:
        shift_out = np.clip(shift_out, *get_range(ret_vdtype))

    mask = canonicalize_mask(mask, x.dtype.lanes)
    shift_out = np.where(mask, shift_out, 0)

    ret = PyVar.zeros(ret_vdtype)
    if x.dtype.bits == 32 and to_h is False:
        ret[::4] = shift_out
    else:
        ret[::2] = shift_out
    return ret


@register_ir_api
def vnsrsr(x, shift, mask=None, out_sign=None, to_h=False):
    """Alias of API "vnsr" with the parameter "saturate" and "with_round" fixed set to True."""
    return vnsr(x, shift, mask, True, out_sign, True, to_h)


@register_ir_api
def _py_vnsrsr(x, shift, mask=None, out_sign=None, to_h=False):
    return _py_vnsr(x, shift, mask, True, out_sign, True, to_h)


@register_ir_api
def vsr(x, shift, mask=None, with_round=False, r=None):
    """Performs a shift by bit to the right of every element of ``x``. Performs arithmetic shift for
    signed and logical shift for unsigned.

    - The inactive elements of result vector are determined by ``r``.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

            x:    -9         8         7         6         5         4         3         2
               11110111  00001000  00000111  00000110  00000101  00000100  00000011  00000010
        shift:     4         3         2         1         0        -1        -2        -3
         mask:     T         T         T         F         T         T         T         T
            z:     0         1         2         3         4         5         6         7

          out = S.vsr(x, shift, mask, r=z)
          out:    -1         1         1         3         5         0         0         0
               11111111  00000001  00000001  00000011  00000101  00000000  00000000  00000000

    Parameters
    ----------
    x : PrimExpr
        The operand.

    shift : Union[PrimExpr, int]
        The shift value. If it is a scalar, it will be automatically broadcast. The negative number
        will be reinterpreted as an unsigned number.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    with_round : Optional[bool]
        Whether the result needs to be rounded or not.

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

        vc = S.vsr(va, vb)
        vc = S.vsr(va, 3)
        vc = S.vsr(va, vb, mask="3T5F")
        vc = S.vsr(va, vb, mask=S.tail_mask(n, 8))
        vc = S.vsr(va, vb, mask=S.tail_mask(n, 8), r=vb)
        vc = S.vsr(va, vb, with_round=True)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vlsr, __vasr, __vlsrr, __vasrr
    """
    x, shift = broadcast_scalar(x, shift)
    assert_vdtype_match(x, shift)
    x_vdtype = x.dtype
    r = canonicalize_r(r, x_vdtype)

    mask = canonicalize_mask(None if r == PARAM_R_MARK else mask, x_vdtype.lanes)
    return tir.call_extern(x_vdtype, "vsr", r, x, shift, mask, with_round)


@register_ir_api
def _py_vsr(x, shift, mask=None, with_round=False, r=None):
    x, shift = broadcast_scalar(x, shift)
    ret = np.around(x * (0.5 ** shift.astype("uint32"))) if with_round else (x >> shift)
    mask = canonicalize_mask(None if r is None else mask, x.dtype.lanes)
    return PyVar(ret, x.dtype, mask, r)


@register_ir_api
def vsl(x, shift, mask=None, saturate=False, r=None):
    """Performs a shift by bit to the left of every element of ``x``.

    - The inactive elements of result vector are determined by ``r``.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

            x:    -9         8         7         6         5         4         3         2
               11110111  00001000  00000111  00000110  00000101  00000100  00000011  00000010
        shift:     4         3         2         1         0        -1        -2        -3
         mask:     T         T         T         T         F         F         T         T
            z:     0         1         2         3         4         5         6         7

          out = S.vsl(x, shift, mask, r=z)
          out:   112        64        28        12         4         5         0         0
               01110000  01000000  00011100  00001100  00000100  00000101  00000000  00000000

    Parameters
    ----------
    x : PrimExpr
        The operand.

    shift : Union[PrimExpr, int]
        The shift value. If it is a scalar, it will be automatically broadcast. The negative number
        will be reinterpreted as an unsigned number.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    saturate : Optional[bool]
        Whether the result needs to be saturated or not.

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

        vc = S.vsl(va, vb)
        vc = S.vsl(va, 3)
        vc = S.vsl(va, vb, mask="3T5F")
        vc = S.vsl(va, vb, mask=S.tail_mask(n, 8))
        vc = S.vsl(va, vb, mask=S.tail_mask(n, 8), saturate=True)
        vc = S.vsl(va, vb, mask=S.tail_mask(n, 8), saturate=True, r=vb)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vsasl, __vlsl
    """
    x, shift = broadcast_scalar(x, shift)
    assert_vdtype_match(x, shift, ignore_sign=True)
    x_vdtype = x.dtype
    if saturate:
        assert x_vdtype.is_int, "Only support input signed dtype if saturate."
    r = canonicalize_r(r, x_vdtype)

    mask = canonicalize_mask(None if r == PARAM_R_MARK else mask, x_vdtype.lanes)
    return tir.call_extern(x_vdtype, "vsl", r, x, shift, mask, saturate)


@register_ir_api
def _py_vsl(x, shift, mask=None, saturate=False, r=None):
    x, shift = broadcast_scalar(x, shift)
    dtype = x.dtype

    ret = x.astype("int64") << np.minimum(shift.astype("uint32"), dtype.bits)
    if saturate:
        ret = np.clip(ret, *get_range(dtype))

    mask = canonicalize_mask(None if r is None else mask, dtype.lanes)
    return PyVar(ret, dtype, mask, r)


@register_ir_api
def vror(x, shift, mask=None):
    """Performs a bit-wise rotate right shift on active elements of ``x``, using the corresponding
    elements of ``shift`` as the shift value.

    - The inactive elements of result vector are undefined.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

            x:     1         2         3        -4        -5         6         7         8
               00000001  00000010  00000011  11111100  11111011  00000110  00000111  00001000
        shift:     5         9        -7         8        -9         0         1         2
         mask:     T         T         T         T         T         T         T         F

          out = S.vror(x, shift, mask)
          out:     8         1       -127       -4        -9         6       -125        ?
               00001000  00000001  10000001  11111100  11110111  00000110  10000011      ?

    Parameters
    ----------
    x : PrimExpr
        The operand.

    shift : Union[PrimExpr, int]
        The shift value. If it is a scalar, it will be automatically broadcast. The negative number
        will be reinterpreted as an unsigned number.

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

        vc = S.vror(va, vb)
        vc = S.vror(va, 3)
        vc = S.vror(va, vb, mask="3T5F")
        vc = S.vror(va, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vror
    """
    x, shift = broadcast_scalar(x, shift)
    assert_vdtype_match(x, shift)
    vdtype = x.dtype
    assert vdtype.is_integer, "Only support integer instruction."
    mask = canonicalize_mask(mask, vdtype.lanes)
    return tir.call_extern(vdtype, "vror", PARAM_R_MARK, x, shift, mask)


@register_ir_api
def _py_vror(x, shift, mask=None):
    x, shift = broadcast_scalar(x, shift)
    dtype = x.dtype

    x = x.astype(dtype.with_uint().element_of)
    shift = shift.astype("uint32") % dtype.bits
    ret = (x >> shift) | (x << (dtype.bits - shift))

    mask = canonicalize_mask(mask, dtype.lanes)
    return PyVar(ret, dtype, mask)


@register_ir_api
def vcls(x, mask=None, out_sign=None):
    """Counts leading sign bits on active elements of ``x``. Leading signs mean continuous bits
    equal to the sign bit from the second higher bit to the lowest bit, excluding the sign bit (the
    highest bit).

    - The inactive elements of result vector are undefined.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x:     0         1         2         5         8        -1        -2        -8
              00000000  00000001  00000010  00000101  00001000  11111111  11111110  11111000
        mask:     T         F         T         T         F         T         F         T

         out = S.vcls(x, mask)
         out:     7         ?         5         4         ?         7         ?         4

    Parameters
    ----------
    x : PrimExpr
        The operand, its data type must be signed.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    out_sign : Optional[str]
        Specify whether the output sign is signed or unsigned. ``None`` means same as ``x``, ``u``
        means unsigned, ``s`` means signed.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32".

    Examples
    --------
    .. code-block:: python

        vr = S.vcls(vx)
        vr = S.vcls(vx, mask="3T5F")
        u_vr = S.vcls(vx, mask="3T5F", out_sign="u")
        vr = S.vcls(vx, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vcls
    """
    assert is_vector(x), "The 1st arg expect a vector."
    x_vdtype = x.dtype
    assert x_vdtype.is_int, f"x expect signed dtype, but got: {x_vdtype}"
    ret_vdtype = change_sign_if_needed(x_vdtype, out_sign)
    mask = canonicalize_mask(mask, x_vdtype.lanes)
    return tir.call_extern(ret_vdtype, "__vcls", PARAM_R_MARK, x, mask)


@register_ir_api
def _py_vcls(x, mask=None, out_sign=None):
    bin_str_list = [f"{item:0{x.dtype.bits}b}" for item in x.view(x.dtype.with_uint().element_of)]
    ret = [len(item) - len(item.lstrip(item[0])) - 1 for item in bin_str_list]
    mask = canonicalize_mask(mask, x.dtype.lanes)
    ret_vdtype = change_sign_if_needed(x.dtype, out_sign)
    return PyVar(ret, ret_vdtype, mask)


@register_ir_api
def clz(x, mask=None):
    """Counts leading zero bits on active elements of ``x``. Leading zeros mean continuous zeros
    from the highest bit to the lowest bit. If the highest bit is 1, the result will be 0 (No
    leading zeros have been found).

    - The scalar situation where ``x`` is a scalar is also supported.
    - The inactive elements of result vector are undefined.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x:     0         1         2         5         8        -1        -2        -8
              00000000  00000001  00000010  00000101  00001000  11111111  11111110  11111000
        mask:     T         F         T         T         F         T         F         T

         out = S.clz(x, mask)
         out:     8         ?         6         5         ?         0         ?         0

    Parameters
    ----------
    x : PrimExpr
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
        "int8/16/32", "uint8/16/32".

    Examples
    --------
    .. code-block:: python

         b = S.clz(3)
        vb = S.clz(va)
        vb = S.clz(va, mask="3T5F")
        vb = S.clz(va, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vclz
    """
    x_dtype = get_dtype(x)
    assert x_dtype.is_integer, "Only support integer input."
    if x_dtype.is_scalar:
        return tir.call_extern(x_dtype, "clz", x)

    # Vector scenario.
    mask = canonicalize_mask(mask, x_dtype.lanes)
    return tir.call_extern(x_dtype, "__vclz", PARAM_R_MARK, x, mask)


@register_ir_api
def _py_clz(x, mask=None):
    x_dtype = get_dtype(x)
    if x_dtype.is_scalar:
        bin_str = np.binary_repr(x, x_dtype.bits)
        return PyVar(len(bin_str) - len(bin_str.lstrip("0")), x_dtype)

    # Vector scenario.
    bin_str_list = [f"{item:0{x.dtype.bits}b}" for item in x.view(x_dtype.with_uint().element_of)]
    ret = [len(item) - len(item.lstrip("0")) for item in bin_str_list]
    mask = canonicalize_mask(mask, x_dtype.lanes)
    return PyVar(ret, x_dtype, mask)


@register_ir_api
def vbrevs(x, mask=None):
    """Performs a binary representation sequance reversal on all active elements in vector x.

    - The inactive elements of result vector are undefined.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x:     0         1         2        5          8        -1       -2         -8
              00000000  00000001  00000010  00000101  00001000  11111111  11111110  11111000
        mask:     F         T         T        T          T         T        T          T

         out = S.vbrevs(x, mask)
         out:     ?       -128       64       -96        16        -1       127        31
                  ?     10000000  01000000  10100000  00010000  11111111  01111111  00011111

    Parameters
    ----------
    x : PrimExpr
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
        "int8/16/32", "uint8/16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vbrevs(va)
        vc = S.vbrevs(va, mask="3T5F")
        vc = S.vbrevs(va, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vbrevs
    """
    assert is_vector(x), "The 1st arg expect a vector."
    x_vdtype = x.dtype
    assert x_vdtype.is_integer_vector, f"The 1st arg expect an integer vector, but got: {x_vdtype}."
    mask = canonicalize_mask(mask, x_vdtype.lanes)
    return tir.call_extern(x_vdtype, "__vbrevs", PARAM_R_MARK, x, mask)


@register_ir_api
def _py_vbrevs(x, mask=None):
    u_dtype = x.dtype.with_uint().element_of
    ret = [int(f"{a:0{x.dtype.bits}b}"[::-1], 2) for a in x.view(u_dtype)]

    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(ret, x.dtype, mask)


@register_ir_api
def vpcnt(x, mask=None, out_sign=None):
    """Counts non-zero bits on active elements of ``x``.

    - The inactive elements of result vector are undefined.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x:     0         1         2         3         4         5         6         7
              00000000  00000001  00000010  00000011  00000100  00000101  00000110  00000111
        mask:     T         F         T         T         F         T         F         T

         out = S.vpcnt(x, mask)
         out:     0         ?         1         2         ?         2         ?         3

    Parameters
    ----------
    x : PrimExpr
        The operand.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    out_sign : Optional[str]
        Specify whether the output sign is signed or unsigned. ``None`` means same as ``x``, ``u``
        means unsigned, ``s`` means signed.

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

        vr = S.vpcnt(vx)
        vr = S.vpcnt(vx, mask="3T5F")
        u_vr = S.vpcnt(vx, mask="3T5F", out_sign="u")
        vr = S.vpcnt(vx, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vpcnt
    """
    assert is_vector(x), "The 1st arg expect a vector."
    x_vdtype = x.dtype
    ret_vdtype = change_sign_if_needed(x_vdtype, out_sign)
    mask = canonicalize_mask(mask, x_vdtype.lanes)
    return tir.call_extern(ret_vdtype, "__vpcnt", PARAM_R_MARK, x, mask)


@register_ir_api
def _py_vpcnt(x, mask=None, out_sign=None):
    ret_vdtype = change_sign_if_needed(x.dtype, out_sign)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    ret = [bin(data).count("1") for data in x.view(x.dtype.with_uint().element_of)]
    return PyVar(ret, ret_vdtype, mask)


__all__ = (
    "vand",
    "vor",
    "vinv",
    "vall",
    "vany",
    "vxor",
    "vnsr",
    "vnsrsr",
    "vsr",
    "vsl",
    "vror",
    "vcls",
    "clz",
    "vbrevs",
    "vpcnt",
)
