# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The permutation part of IR APIs."""
import numpy as np
from tvm import tir, DataType
from ...utils import hw_native_vdtype, is_hw_native_vdtype
from ..pysim import PyVar
from .base import register_ir_api, canonicalize_mask
from .utils import PARAM_R_MARK, broadcast_scalar, assert_vdtype_match, within_range
from .utils import is_scalar_prim_expr, is_integer_scalar, assert_lanes_valid, is_vector
from .utils import is_vector_or_mask, VALID_PARTS, assert_not_flexible_width_vector
from .utils import assert_neither_flexible_nor_multiple_width_vector


@register_ir_api
def vconcat(inps, part="all"):
    """Concats ``inps`` with elements according to the value of parameter ``part``.

    - The feature Multiple Width Vector is supported.

    .. code-block::

           x(i32x8): x0  x1  x2  x3  x4  x5  x6  x7
           y(i32x8): y0  y1  y2  y3  y4  y5  y6  y7
           z(i32x8): z0  z1  z2  z3  z4  z5  z6  z7


        out = S.vconcat((x, y))
        out(i32x16): x0  x1  x2  x3  x4  x5  x6  x7  y0  y1  y2  y3  y4  y5  y6  y7

        out = S.vconcat((x, y, z))
        out(i32x24): x0  x1  x2  x3  x4  x5  x6  x7  y0  ...  y7  z0  ...  z7

        out = S.vconcat((x, y), "low")
        out(i32x8): x0  x1  x2  x3  y0  y1  y2  y3

        out = S.vconcat((x, y), "high")
        out(i32x8): x4  x5  x6  x7  y4  y5  y6  y7

        out = S.vconcat((x, y), "even")
        out(i32x8): x0  x2  x4  x6  y0  y2  y4  y6

        out = S.vconcat((x, y), "odd")
        out(i32x8): x1  x3  x5  x7  y1  y3  y5  y7

    Parameters
    ----------
    inps : Union[List[PrimExpr, int, float], Tuple[PrimExpr, int, float]]
        The operands.

    part : str
        Used to specify which part elements to be selected.

        - **all:** Represent all data.
        - **low, high, even, odd:** Represent the corresponding half data.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vconcat((va, vb), "low")
        vc = S.vconcat((va, 3), "high")

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vextl, __vexth, __vexte, __vexto.
    """
    msg = f'The arg "inps" expect one of (tuple, list) and len(inps) > 1, but got: "{inps}".'
    assert isinstance(inps, (tuple, list)) and len(inps) > 1, msg
    inps = broadcast_scalar(*inps)
    assert_vdtype_match(*inps)
    msg = f'The arg "part" expect one of {VALID_PARTS[:5]}, but got: "{part}".'
    assert part in VALID_PARTS[:5], msg

    x_vdtype = DataType(inps[0].dtype)
    assert_not_flexible_width_vector(x_vdtype)
    ret_lanes = x_vdtype.lanes * len(inps)
    ret_lanes = ret_lanes if part == "all" else ret_lanes // 2
    ret_vdtype = x_vdtype.with_lanes(ret_lanes)
    return tir.call_extern(ret_vdtype, "vconcat", *inps, part)


@register_ir_api
def _py_vconcat(inps, part="all"):
    inps = broadcast_scalar(*inps)
    x_dtype = inps[0].dtype
    half_lanes = x_dtype.lanes // 2
    ret_lanes = x_dtype.lanes * len(inps)
    ret_lanes = ret_lanes if part == "all" else ret_lanes // 2
    ret_vdtype = x_dtype.with_lanes(ret_lanes)

    if part == "low":
        ret = np.concatenate([x[:half_lanes] for x in inps])
    elif part == "high":
        ret = np.concatenate([x[half_lanes:] for x in inps])
    elif part == "even":
        ret = np.concatenate([x[::2] for x in inps])
    elif part == "odd":
        ret = np.concatenate([x[1::2] for x in inps])
    else:
        ret = np.concatenate(inps)

    return PyVar(ret, ret_vdtype)


@register_ir_api
def vsplit(x):
    """Splits ``x`` to multiple parts evenly according to the hardware native vector types.

    - The feature Multiple Width Vector is supported.

    .. code-block::

          x(i32x16): x0  x1   x2   x3   x4   x5   x6   x7  x8  x9  x10  x11  x12  x13  x14  x15

        out0, out1 = S.vsplit(x)
        out0(i32x8): x0  x1   x2   x3   x4   x5   x6   x7
        out1(i32x8): x8  x9  x10  x11  x12  x13  x14  x15

    Parameters
    ----------
    x : PrimExpr
        The operand.

    Returns
    -------
    ret : Tuple[PrimExpr]
        The result expressions.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32".

    Examples
    --------
    .. code-block:: python

        vc0, vc1 = S.vsplit(vx_fp32x16)
        vc0, vc1, vc2 = S.vsplit(vx_i32x24)

    See Also
    --------
    - :doc:`../../language_basics/flexible_width_and_multiple_width_vector`
    """
    assert is_vector(x), "The 1st arg expect a vector."
    x_vdtype = DataType(x.dtype)
    assert_not_flexible_width_vector(x_vdtype)
    hw_vdtype = hw_native_vdtype(x_vdtype)
    part_cnt = x_vdtype.lanes // hw_vdtype.lanes
    assert part_cnt != 1, "Redundant useless call."
    return tuple(tir.call_extern(hw_vdtype, "vsplit", x, i) for i in range(part_cnt))


@register_ir_api
def _py_vsplit(x):
    hw_vdtype = hw_native_vdtype(x.dtype)
    hw_lanes = hw_vdtype.lanes
    part_cnt = x.dtype.lanes // hw_lanes
    return tuple(PyVar(x[i * hw_lanes : (i + 1) * hw_lanes], hw_vdtype) for i in range(part_cnt))


@register_ir_api
def vzip(x, y, part="all"):
    """Selects some elements from ``x`` and ``y``, rearranges in an interleave alternate sequence,
    and then returns. The selected elements are according to the value of parameter ``part``.

    - The mask situation where both ``x`` and ``y`` are mask is also supported.
    - The feature Flexible Width Vector is supported only when ``part`` is ``all``.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x(i32x8): x0  x1  x2  x3  x4  x5  x6  x7
           y(i32x8): y0  y1  y2  y3  y4  y5  y6  y7

        out = S.vzip(x, y)
        out(i32x16): x0  y0  x1  y1  x2  y2  x3  y3  x4  y4  x5  y5  x6  y6  x7  y7

         out = S.vzip(x, y, "low")
         out(i32x8): x0  y0  x1  y1  x2  y2  x3  y3

         out = S.vzip(x, y, "high")
         out(i32x8): x4  y4  x5  y5  x6  y6  x7  y7

         out = S.vzip(x, y, "even")
         out(i32x8): x0  y0  x2  y2  x4  y4  x6  y6

         out = S.vzip(x, y, "odd")
         out(i32x8): x1  y1  x3  y3  x5  y5  x7  y7

    Parameters
    ----------
    x, y : Union[PrimExpr, int, float, bool]
        The operands.

    part : Optional[str]
        Used to specify which part elements to be selected.

        - **all:** Represent all data.
        - **low, high, even, odd:** Represent the corresponding half data.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32", "bool8/16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vzip(va, vb)
        vc = S.vzip(va, vb, "low")
        vc = S.vzip(va, 3)
        vc = S.vzip(va, 3, "high")
        vc = S.vzip(va, True, "even")

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vzipl, __vziph, __vzipe, __vzipo
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    x_vdtype = DataType(x.dtype)

    msg = f'The arg "part" expect one of {VALID_PARTS[:5]}, but got: "{part}".'
    assert part in VALID_PARTS[:5], msg
    if part == "all":
        ret_vdtype = x_vdtype.with_lanes(x_vdtype.lanes * 2)
    else:
        assert_not_flexible_width_vector(x_vdtype)
        ret_vdtype = x_vdtype

    return tir.call_extern(ret_vdtype, "vzip", x, y, part)


@register_ir_api
def _py_vzip(x, y, part="all"):
    x, y = broadcast_scalar(x, y)

    if part == "all":
        ret = PyVar.zeros(x.dtype.with_lanes(x.dtype.lanes * 2))
        ret[::2], ret[1::2] = x, y
        return ret

    half_lanes = x.dtype.lanes // 2
    ret = PyVar.zeros(x.dtype)
    if part == "low":
        ret[::2], ret[1::2] = x[:half_lanes], y[:half_lanes]
    elif part == "high":
        ret[::2], ret[1::2] = x[half_lanes:], y[half_lanes:]
    elif part == "even":
        ret[::2], ret[1::2] = x[::2], y[::2]
    else:
        ret[::2], ret[1::2] = x[1::2], y[1::2]
    return ret


@register_ir_api
def vcompt(x, mask):
    """Reads active elements from ``x``, packs them into the lowest-numbered elements of result
    vector.

    - The remaining upper elements of result vector are set to zero.

    .. code-block::

           x: x0  x1  x2  x3  x4  x5  x6  x7
        mask:  T   T   F   F   T   T   T   F

         out = S.vcompt(x, mask)
         out: x0  x1  x4  x5  x6   0   0   0

    Parameters
    ----------
    x : PrimExpr
        The operands.

    mask : Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vcompt(vx, mask="3T5F")
        vc = S.vcompt(vx, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vcompt
    """
    assert is_vector(x), "The 1st arg expect a vector."
    x_vdtype = DataType(x.dtype)
    assert_neither_flexible_nor_multiple_width_vector(x_vdtype)
    mask = canonicalize_mask(mask, x_vdtype.lanes)
    return tir.call_extern(x_vdtype, "__vcompt", x, mask)


@register_ir_api
def _py_vcompt(x, mask):
    mask = canonicalize_mask(mask, x.dtype.lanes)

    ret = PyVar.zeros(x.dtype)
    j = 0
    for i in range(x.dtype.lanes):
        if mask[i]:
            ret[j] = x[i]
            j += 1
    return ret


@register_ir_api
def vcompc(x, y, mask):
    """Reads active elements from ``x``, packs them into the lowest-numbered elements of result
    vector.

    - The remaining upper elements of result vector are set to the lowest-numbered elements of
      ``y``.

    .. code-block::

           x: x0  x1  x2  x3  x4  x5  x6  x7
           y: y0  y1  y2  y3  y4  y5  y6  y7
        mask:  T   T   F   T   F   T   F   F

         out = S.vcompc(x, y, mask)
         out: x0  x1  x3  x5  y0  y1  y2  y3

    Parameters
    ----------
    x, y : Union[PrimExpr, int, float]
        The inputs to be packed.

    mask : Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vcompc(va, vb, mask="3T5F")
        vc = S.vcompc(va, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vcompc
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    x_vdtype = DataType(x.dtype)
    assert_neither_flexible_nor_multiple_width_vector(x_vdtype)
    mask = canonicalize_mask(mask, x_vdtype.lanes)
    return tir.call_extern(x_vdtype, "__vcompc", x, y, mask)


@register_ir_api
def _py_vcompc(x, y, mask):
    x, y = broadcast_scalar(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)

    ret = PyVar.zeros(x.dtype)
    j = 0
    for i in range(x.dtype.lanes):
        if mask[i]:
            ret[j] = x[i]
            j += 1

    for i in range(x.dtype.lanes - j):
        ret[j + i] = y[i]
    return ret


@register_ir_api
def vrevs(x):
    """Reverses the order of all elements in ``x``.

    - The mask situation where ``x`` is a mask is also supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

          x: x0  x1  x2  x3  x4  x5  x6  x7

        out = S.vrevs(x)
        out: x7  x6  x5  x4  x3  x2  x1  x0

    Parameters
    ----------
    x : PrimExpr
        The operands.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32", "bool8/16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vrevs(vx)
        mask_revs = S.vrevs(vx > 0)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vrevs, __vprevs
    """
    assert is_vector_or_mask(x), "The 1st arg expect a vector or a mask."
    assert_not_flexible_width_vector(x.dtype)
    return tir.call_extern(x.dtype, "vrevs", x)


@register_ir_api
def _py_vrevs(x):
    return PyVar(np.flip(x), x.dtype)


@register_ir_api
def vsel(x, y, mask=None):
    """Select active elements from ``x`` and inactive elements from ``y``.

    - The mask situation where both ``x`` and ``y`` are mask is also supported.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

           x: x0  x1  x2  x3  x4  x5  x6  x7
           y: y0  y1  y2  y3  y4  y5  y6  y7
        mask:  T   T   F   F   T   T   T   F

         out = S.vsel(x, y, mask)
         out: x0  x1  y2  y3  x4  x5  x6  y7

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
        "int8/16/32", "uint8/16/32", "float16/32", "bool8/16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vsel(va, vb)
        vc = S.vsel(va, 3)
        vc = S.vsel(3, vb)
        vc = S.vsel(va, vb, mask="3T5F")
        vc = S.vsel(va, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vsel
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    vdtype = DataType(x.dtype)
    mask = canonicalize_mask(mask, vdtype.lanes)
    return tir.call_extern(vdtype, "__vsel", x, y, mask)


@register_ir_api
def _py_vsel(x, y, mask=None):
    x, y = broadcast_scalar(x, y)
    mask = canonicalize_mask(mask, x.dtype.lanes)
    return PyVar(np.where(mask, x, y), x.dtype)


@register_ir_api
def vshfl(x, shift):
    """Performs a rotate shift by element from high to low direction in vector ``x``, with the
    shift number of the value of ``shift``.

    - The feature Multiple Width Vector is supported.

    .. code-block::

            # shift direction:   <----
            x: x0  x1  x2  x3  x4  x5  x6  x7
        shift: 2

          out = S.vshfl(x, shift)
          out: x2  x3  x4  x5  x6  x7  x0  x1

    Parameters
    ----------
    x : PrimExpr
        The operand, should be a vector.

    shift : Union[PrimExpr, int]
        The shift value, should be a scalar.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vshfl(va, scalar_var)
        vc = S.vshfl(va, 3)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vshfl
    """
    assert is_vector(x), "The 1st arg expect a vector."
    assert is_integer_scalar(shift), 'The arg "shift" expects an integer scalar.'
    if isinstance(shift, int):
        assert_not_flexible_width_vector(x.dtype)
        if not is_hw_native_vdtype(x.dtype):
            return vsldl(x, x, shift)
    else:
        assert_neither_flexible_nor_multiple_width_vector(x.dtype)
    return tir.call_extern(x.dtype, "__vshfl", x, shift)


@register_ir_api
def _py_vshfl(x, shift):
    lanes = x.dtype.lanes
    assert 0 <= shift < lanes, f'The arg "shift" expect in range [0, {lanes}), but got: "{shift}".'
    return PyVar(np.concatenate([x[shift:], x[:shift]]), x.dtype)


@register_ir_api
def vbcast(x, mask=None, lanes=None):
    """Performs a broadcast operation on a scalar with dtype to generate a vector.

    - The inactive elements of result vector are undefined.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

            x(i32): 3
              mask: T  T  T  T  F  F  F  F

        out = S.vbcast(S.i32(3))
        out(i32x8): 3  3  3  3  ?  ?  ?  ?

    Parameters
    ----------
    x : PrimExpr
        The operand.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    lanes : Optional[int]
        The lanes of result vector dtype. If omitted, will be automatically determined based on
        the type of input value.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vbcast(S.i16(5))
        vc = S.vbcast(S.fp16(1.23))
        vc = S.vbcast(S.fp32(3.14159), mask="3T5F")
        vc = S.vbcast(S.u32(1), mask="T7F")
        vc = S.vbcast(S.u8(200) mask=S.tail_mask(n, 8))
        vc = S.vbcast(S.i16(5), lanes=16)
        vc = S.vbcast(x > 0, lanes=16)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vbcast
    """
    assert is_scalar_prim_expr(x), "The 1st arg expect a scalar that contain type information."
    x_dtype = DataType(x.dtype)

    if x_dtype.is_bool:
        assert mask is None, 'The arg "mask" can not supported when broadcasting a boolean.'
        assert lanes is not None, 'The arg "lanes" must be provided when broadcasting a boolean.'
        ret_vdtype = x_dtype
    else:
        ret_vdtype = hw_native_vdtype(x.dtype)

    if lanes is not None:
        assert_lanes_valid(lanes)
        ret_vdtype = ret_vdtype.with_lanes(lanes)

    mask = canonicalize_mask(mask, ret_vdtype.lanes)
    return tir.call_extern(ret_vdtype, "__vbcast", PARAM_R_MARK, x, mask)


@register_ir_api
def _py_vbcast(x, mask=None, lanes=None):
    if isinstance(x, bool):  # Scalar bool isn't represented by PyVar in PySim.
        ret_vdtype = DataType(f"boolx{lanes}")
    else:
        ret_vdtype = hw_native_vdtype(x.dtype) if lanes is None else x.dtype.with_lanes(lanes)

    ret = np.broadcast_to(x, ret_vdtype.lanes)
    mask = canonicalize_mask(mask, ret_vdtype.lanes)
    return PyVar(ret, ret_vdtype, mask)


@register_ir_api
def vsldl(x, y, shift):
    """Shift left a unsigned immediate value shift for ``x``, and pads the shift remained space
    with ``y``.

    - The feature Multiple Width Vector is supported.

    .. code-block::

            # shift direction:   <----
            x: x0  x1  x2  x3  x4  x5  x6  x7
            y: y0  y1  y2  y3  y4  y5  y6  y7
        shift: 3

          out = S.vsldl(x, y, shift)
          out: x3  x4  x5  x6  x7  y0  y1  y2

    Parameters
    ----------
    x, y : Union[PrimExpr, int, float]
        The operands. If either one is a scalar, it will be automatically broadcast. The ``x`` and
        ``y`` should be of the same type.

    shift : int
        The unsigned immediate value for the shift left operation.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vsldl(va, vb, shift=2)
        vc = S.vsldl(va, 3, shift=2)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vsldl
    """
    x, y = broadcast_scalar(x, y)
    assert_vdtype_match(x, y)
    x_vdtype = DataType(x.dtype)
    assert_not_flexible_width_vector(x_vdtype)

    assert isinstance(shift, int), 'The arg "shift" expect a constant integer scalar.'
    msg = f'The arg "shift" expect in range [0, {x_vdtype.lanes}), but got: "{shift}".'
    assert within_range(shift, low=0, high=x_vdtype.lanes), msg
    return tir.call_extern(x_vdtype, "__vsldl", x, y, shift)


@register_ir_api
def _py_vsldl(x, y, shift):
    x, y = broadcast_scalar(x, y)
    return PyVar(np.concatenate([x[shift:], y[:shift]]), x.dtype)


@register_ir_api
def vtbl(table, indices):
    """Constructs ``table`` with 2 ~ 4 vector (a, b), (a, b, c) or (a, b, c, d), reads each element
    of vector ``indices`` as an index to select the element from ``table``, and places the indexed
    element in the corresponding element of result vector. If an index value is >= the element
    count of the ``table``, then places 0 in the corresponding element of result vector.

    .. code-block::

              a: t0  t1  t2  t3  t4  t5  t6  t7
              b: t8  t9  t10 t11 t12 t13 t14 t15
              c: t16 t17 t18 t19 t20 t21 t22 t23
              d: t24 t25 t26 t27 t28 t29 t30 t31
        indices: 1   1   5   7   3   10  99  100

            out = S.vtbl((a, b, c, d), indices)
            out: t1  t1  t5  t7  t3  t10 0   0

    Parameters
    ----------
    table : Union[List[PrimExpr], Tuple[PrimExpr]]
        A list of vector to indicate a table.

    indices : PrimExpr
        The indices of table.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32".

    Examples
    --------
    .. code-block:: python

        vout = S.vtbl((va, vb, vc, vd), index_vector)
        vout = S.vtbl((va, vb, vc), index_vector)
        vout = S.vtbl((va, vb), index_vector)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vtbl, __vperm
    """
    assert isinstance(table, (tuple, list)), 'The arg "table" expect a tuple or list.'
    msg = f'The length of the arg "table" expect in range [2, 4], but got: "{len(table)}".'
    assert 2 <= len(table) <= 4, msg
    assert is_vector(indices), 'The arg "indices" expect a vector.'

    table = broadcast_scalar(*table)
    assert_vdtype_match(*table)
    table_vdtype = DataType(table[0].dtype)
    assert_neither_flexible_nor_multiple_width_vector(table_vdtype)
    indices_vdtype = DataType(indices.dtype)

    assert indices_vdtype.is_integer, 'The arg "indices" expects an integer vector.'
    msg = f'The bits mismatch: "table": "{table_vdtype}" vs. "indices": "{indices_vdtype}".'
    assert table_vdtype.bits == indices_vdtype.bits, msg
    return tir.call_extern(table_vdtype, "vtbl", *table, indices)


@register_ir_api
def _py_vtbl(table, indices):
    table = broadcast_scalar(*table)
    table = np.concatenate(table)

    table_indices = list(range(len(table)))
    r = [table[idx] if idx in table_indices else 0 for idx in indices]
    return PyVar(r, table[0].dtype)


@register_ir_api
def vreplic(x, index=0):
    """Uses the scalar ``index`` to choose one element from ``x``, then replicates all elements of
    result vector by this element.

    - The feature Multiple Width Vector is supported.

    .. code-block::

            x: x0  x1  x2  x3  x4  x5  x6  x7
        index: 3

          out = S.vreplic(x, index)
          out: x3  x3  x3  x3  x3  x3  x3  x3

    Parameters
    ----------
    x : PrimExpr
        The operand.

    index : Optional[Union[PrimExpr, int]]
        The index should be a scalar.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32", "float16/32".

    Examples
    --------
    .. code-block:: python

        vc = S.vreplic(va, scalar_var)
        vc = S.vreplic(va, 3)
        vc = S.vreplic(va)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vreplic
    """
    assert is_vector(x), "The 1st arg expect a vector."
    assert_not_flexible_width_vector(x.dtype)
    if is_hw_native_vdtype(x.dtype):
        assert is_integer_scalar(index), 'The arg "index" expects an integer scalar.'
    else:
        assert isinstance(index, int), 'The arg "index" expects an integer constant value.'
    return tir.call_extern(x.dtype, "__vreplic", x, index)


@register_ir_api
def _py_vreplic(x, index=0):
    lanes = x.dtype.lanes
    assert 0 <= index < lanes, f'The arg "index" expect in range [0, {lanes}), but got: "{index}".'
    return PyVar([x[index]] * lanes, x.dtype)


__all__ = (
    "vconcat",
    "vsplit",
    "vzip",
    "vcompt",
    "vcompc",
    "vrevs",
    "vsel",
    "vshfl",
    "vbcast",
    "vsldl",
    "vtbl",
    "vreplic",
)
