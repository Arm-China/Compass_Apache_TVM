# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""The conversion part of IR APIs."""
import numpy as np
from tvm import tir, get_range
from tvm.script import tir as T
from ...utils import ALIAS2ELEMENT_DTYPE, resolve_dtype_alias, double_elem_width
from ...utils import is_hw_native_vdtype
from ..pysim import PyVar, PyPointer
from .base import register_ir_api
from .utils import is_scalar, is_vector, get_dtype, VALID_PARTS, assert_not_flexible_width_vector
from .utils import assert_neither_flexible_nor_multiple_width_vector
from .permutation import vbcast


_ALIAS2DTYPE = {
    "i8x32": "int8x32",
    "u8x32": "uint8x32",
    "i16x16": "int16x16",
    "i16x32": "int16x32",
    "u16x16": "uint16x16",
    "u16x32": "uint16x32",
    "i32x8": "int32x8",
    "i32x16": "int32x16",
    "i32x32": "int32x32",
    "u32x8": "uint32x8",
    "u32x16": "uint32x16",
    "u32x32": "uint32x32",
    "fp16x16": "float16x16",
    "fp16x32": "float16x32",
    "fp32x8": "float32x8",
    "fp32x16": "float32x16",
    "bf16x16": "bfloat16x16",
}
_ALIAS2DTYPE.update(ALIAS2ELEMENT_DTYPE)


def _get_cast_return_type(to_dtype, expect_lanes, msg):
    assert to_dtype.lanes in (1, expect_lanes), msg
    return to_dtype.with_lanes(expect_lanes)


def _check_unsupported_direct_cast(from_dtype, to_dtype):
    if (from_dtype.is_floating16 and to_dtype.is_integer and to_dtype.bits < 32) or (
        from_dtype.is_integer and from_dtype.bits < 32 and to_dtype.is_floating16
    ):
        raise ValueError(
            f'Unsupported cast directly from "{from_dtype}" to "{to_dtype}", because there isn\'t '
            "any hardware conversion instructions between float16x16/bfloat16x16 and narrower bits"
            " integer, so you need to implement it manually through 32-bit and multiple invocations"
            ' of "cast".'
        )

    unsupported_direct_cast = (
        ("uint32", "float32"),
        ("float32", "uint32"),
        ("uint32", "float16"),
        ("uint32", "bfloat16"),
        ("bfloat16", "uint32"),
    )
    if (from_dtype.element_of, to_dtype.element_of) in unsupported_direct_cast:
        raise ValueError(
            f'Unsupported cast directly from "{from_dtype}" to "{to_dtype}" without precision '
            'loss, you need to implement it manually through multiple invocations of "cast".'
        )


def _update_saturate_with_check(from_dtype, to_dtype, saturate):
    if to_dtype.is_floating:
        return "None"

    # From here, the target data type is integer.
    if from_dtype.is_floating:
        # 1. Cast from float to int32, e.g., fp16x16 -> i32x8, fp32x8 -> i32x8.
        if to_dtype.is_int32:
            msg = f'Unsupported cast from "{from_dtype}" to "{to_dtype}" without saturation.'
            assert saturate in (None, True), msg
            return True

        # 2. Cast from float to other integer types, e.g., fp16x16 -> u32x8, fp32x8 -> i8x32.
        msg = f'Currently can\'t set "saturate" when casting from "{from_dtype}" to "{to_dtype}".'
        assert saturate is None, msg
        return "None"

    # From here, both the source and the target data type are integer.
    return False if saturate is None else saturate


@register_ir_api
def cast(x, dtype, part="all", saturate=None):
    """Converts the given expression or value to the specified type.

    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    x : Union[tir.PrimExpr, Literal["inf", "-inf", "nan"], int, float, list, tuple]
        The expression or value that needs to be cast.

    dtype : Union[str, DataType]
        The target data type. If it is set to a scalar one when the given expression or value is a
        vector, it will be changed to a vector one with the lanes of the given expression or value
        automatically.

    part : Optional[str]
        Only used for the vector conversion, used to specify which part data of the given expression
        or value needs to be converted.

        - **all:** Represent all data.
        - **low, high, even, odd:** Represent the corresponding half data.
        - **ll, lh, hl, hh:** Represent the corresponding quarter data.

    saturate : Optional[bool]
        Whether the result needs to be saturated or not. Only used when the target data type is
        integer, and its range does not complete contain that of the source data type, e.g.,
        ``i8`` -> ``u16``, ``(i32x8, i32x8)`` -> ``i16x16``. ``None`` means auto set according to
        best performance, i.e., when casting floating to integer, ``True`` for
        ``float16``/``float32``/``bfloat16`` to ``int32`` stage, ``False`` for ``int32`` to the
        target data type stage, ``False`` when casting in other situations.

    Returns
    -------
    ret : tir.PrimExpr
        The result expression with the needed type.

    Examples
    --------
    .. code-block:: python

        scalar_b = S.cast(scalar_var, dtype)
        fp32_100 = S.cast(100, "float32")
        vb = S.cast(va, dtype)
        vindex0_u = S.cast(vindex0, "uint16x16")
        inp1_fp32_l, inp1_fp32_h = S.cast(inp1, "float32", "low"), S.cast(inp1, "float32", "high")
        va_fp32_e = S.cast(va, "fp32", part="even")
        va_fp32_o = S.cast(va, "fp32", part="odd")
        va_fp16x16 = S.cast((va_fp32x8, va_fp32x8), "fp16")
        vb = S.cast(va, dtype, saturate=True)

    See Also
    --------
    - :doc:`../../language_basics/types`
    """
    if isinstance(x, str):
        msg = f'The 1st arg expect one of ("inf", "-inf", "nan"), but got: "{x}".'
        assert x.lower() in ("inf", "-inf", "nan"), msg
        x = float(x)
    elif isinstance(x, np.ndarray):
        x = x.flatten().tolist()

    to_dtype = resolve_dtype_alias(dtype)
    to_bits, to_lanes = to_dtype.bits, to_dtype.lanes
    msg = f'The arg "part" expect one of {VALID_PARTS}, but got: "{part}".'
    assert part in VALID_PARTS, msg
    msg = f'The arg "saturate" expect one of (None, True, False), but got: "{saturate}".'
    assert saturate in (None, True, False), msg

    # 1. Scalar and broadcast scenarios.
    if is_scalar(x):
        assert saturate is None, 'Currently can\'t set "saturate" when the 1st arg is a scalar.'
        ret = tir.cast(x, to_dtype.element_of)
        return ret if to_dtype.is_scalar else vbcast(ret, lanes=to_lanes)

    # 2. The scenarios that target data type and the given expression are all vector.
    if isinstance(x, (list, tuple)):
        x_len = len(x)
        assert x_len > 1, "The length of the 1st arg must be > 1 when it's a tuple or list."

        # 2.1 The vector literal used to initialize a vector variable.
        if all(is_scalar(item) for item in x):
            assert saturate is None, 'Currently can\'t set "saturate" when creating vector literal.'
            msg = f'The length of the 1st arg expect "{to_lanes}", but got: "{x_len}".'
            ret_vdtype = _get_cast_return_type(to_dtype, x_len, msg)
            return tir.call_intrin(ret_vdtype, "tir.vector_literal", *x)

        # 2.2 Cast to narrower bits with merge, e.g., (fp32x8, fp32x8) -> fp16x16,
        #     (i32x8, i32x8) -> u8x32, (u32x8, u32x8, u32x8) -> i8x32.
        assert all(isinstance(item, tir.PrimExpr) for item in x), "All items must be variable."
        x0_dtype = x[0].dtype

        msg = "The data type of all items must be same and must be vector."
        assert x0_dtype.is_vector and all(x[0].dtype == item.dtype for item in x), msg
        msg = "Only cast to 1/2 or 1/4 narrower bits allow multiple inputs."
        assert x0_dtype.bits // to_bits in (2, 4), msg
        assert part == "all", 'The arg "part" only support "all" when cast to narrower bits.'
        _check_unsupported_direct_cast(x0_dtype, to_dtype)
        saturate = _update_saturate_with_check(x0_dtype, to_dtype, saturate)

        ret_vdtype = to_dtype.with_lanes(x0_dtype.lanes * x_len) if to_lanes == 1 else to_dtype
        msg = f'Unsupported cast from "{(x0_dtype,) * x_len}" to "{ret_vdtype}".'
        assert all(is_hw_native_vdtype(dtype) for dtype in (x0_dtype, ret_vdtype)), msg
        return tir.call_extern(ret_vdtype, "vcast", part, saturate, *x)

    # From here, the input of cast must be a single vector expression.
    assert is_vector(x), "The 1st arg expect a vector."
    from_dtype = x.dtype
    from_bits, from_lanes = from_dtype.bits, from_dtype.lanes
    _check_unsupported_direct_cast(from_dtype, to_dtype)
    saturate = _update_saturate_with_check(from_dtype, to_dtype, saturate)
    msg = f'Unsupported cast from "{from_dtype}" to "{to_dtype}" with part "{part}".'

    if from_bits == to_bits:
        assert part == "all", 'The arg "part" only support "all" when cast to same bits.'
        assert to_lanes in (1, from_lanes), msg

        # 2.3 Redundant cast, e.g., i8x32 -> i8x32.
        if from_dtype.type_code == to_dtype.type_code:
            return x

        # 2.4 Cast to same bits, e.g., i8 -> u8, i32 -> fp32.
        expect_lanes = from_lanes

    elif from_bits < to_bits:
        # 2.5 Cast to wider bits, e.g., i8 -> i32, fp16 -> fp32.
        if part == "all":  # e.g., i8x32 -> i32x32, fp16x16 -> i32x16.
            expect_lanes = from_lanes
        else:
            assert is_hw_native_vdtype(from_dtype), msg

            if part in ("low", "high", "even", "odd"):  # e.g., i8x32 -> i16x16.
                expect_lanes = from_lanes // 2
                assert to_bits // from_bits == 2, msg
            else:  # ("ll", "lh", "hl", "hh"), e.g., i8x32 -> i32x8, u8x32 -> fp32x8
                expect_lanes = from_lanes // 4
                assert to_bits // from_bits == 4, msg
    else:
        # 2.6 Cast to narrower bits without merge, e.g., i32 -> i8, fp32 -> fp16.
        assert part == "all", 'The arg "part" only support "all" when cast to narrower bits.'
        # The other value besides "all" is meaningless for this situation, e.g., i32x8 -> i8x4.
        if to_lanes in (1, from_lanes):
            # The lanes is same, e.g., i32x55 -> i8x55, i32x8 -> i8x8, i16x16 -> i8x16.
            expect_lanes = from_lanes
        else:
            # The lanes is different, e.g., i32x8 -> i8x32, u32x8 -> fp16x16.
            assert all(is_hw_native_vdtype(dtype) for dtype in (from_dtype, to_dtype)), msg
            expect_lanes = to_lanes

    ret_vdtype = _get_cast_return_type(to_dtype, expect_lanes, msg)
    return tir.call_extern(ret_vdtype, "vcast", part, saturate, x)


def _saturate_if_needed(x, to_dtype, saturate):
    if to_dtype.is_floating:
        return x

    # From here, the target data type is integer.
    from_dtype = get_dtype(x)
    if from_dtype.is_floating:
        # Set `0` to the same data type as `ret` to avoid type promotion issues.
        ret = np.where(np.isnan(x), np.array(0, dtype=x.dtype), x)
        # Here will promote to "float64" automatically, so it's safe.
        return np.clip(ret, *get_range("int32"))

    # From here, both the source and the target data type are integer.
    return np.clip(x, *get_range(to_dtype)) if saturate is True else x


@register_ir_api
def _py_cast(x, dtype, part="all", saturate=None):
    if isinstance(x, str):
        x = float(x)
    elif isinstance(x, np.ndarray):
        x = x.flatten().tolist()

    to_dtype = resolve_dtype_alias(dtype)
    to_lanes = to_dtype.lanes

    # 1. Scalar and broadcast scenarios.
    if is_scalar(x):
        ret = _saturate_if_needed(x, to_dtype, saturate)

        if to_dtype.is_vector:
            ret = np.broadcast_to(ret, to_lanes)
        return PyVar(ret, to_dtype)

    # 2. The scenarios that target data type and the given expression are all vector.
    if isinstance(x, (list, tuple)):
        if all(is_scalar(item) for item in x):
            # 2.1 The vector literal used to initialize a vector.
            return PyVar(x, to_dtype.with_lanes(len(x)))

        # 2.2 Cast to narrower bits with merge, e.g., (fp32x8, fp32x8) -> fp16x16,
        #     (i32x8, i32x8) -> u8x32, (u32x8, u32x8, u32x8) -> i8x32.
        ret = np.concatenate(x)

        ret_vdtype = to_dtype.with_lanes(len(ret)) if to_lanes == 1 else to_dtype
        if len(ret) != ret_vdtype.lanes:
            ret = np.resize(ret, ret_vdtype.lanes)

        return PyVar(_saturate_if_needed(ret, to_dtype, saturate), ret_vdtype)

    # From here, the input of cast must be a single vector expression.
    from_dtype = x.dtype
    from_lanes = from_dtype.lanes
    ret = x.value

    if from_dtype.bits == to_dtype.bits:
        # 2.3 Redundant cast, e.g., i8x32 -> i8x32.
        if from_dtype.type_code == to_dtype.type_code:
            return x

        # 2.4 Cast to same bits, e.g., i8 -> u8, i32 -> fp32.
        expect_lanes = from_lanes
    elif from_dtype.bits < to_dtype.bits:
        # 2.5 Cast to wider bits, e.g., i8 -> i32, fp16 -> fp32.
        if part == "all":  # e.g., i8x32 -> i32x32, fp16x16 -> i32x16.
            expect_lanes = from_lanes
        elif part in ("low", "high", "even", "odd"):  # e.g., i8x32 -> i16x16.
            expect_lanes = from_lanes // 2
            if part == "low":
                ret = x[:expect_lanes]
            elif part == "high":
                ret = x[expect_lanes:]
            elif part == "even":
                ret = x[::2]
            else:
                ret = x[1::2]
        else:  # ("ll", "lh", "hl", "hh"), e.g., i8x32 -> i32x8, u8x32 -> fp32x8
            expect_lanes = from_lanes // 4
            if part == "ll":
                ret = x[:expect_lanes]
            elif part == "lh":
                ret = x[expect_lanes : expect_lanes * 2]
            elif part == "hl":
                ret = x[expect_lanes * 2 : expect_lanes * 3]
            else:
                ret = x[expect_lanes * 3 :]
    else:
        # 2.6 Cast to narrower bits without merge, e.g., i32 -> i8, fp32 -> fp16.
        if to_lanes in (1, from_lanes):
            # The lanes is same, e.g., i32x55 -> i8x55, i32x8 -> i8x8, i16x16 -> i8x16.
            expect_lanes = from_lanes
        else:
            # The lanes is different, e.g., i32x8 -> i8x32, i32x8 -> fp16x16.
            expect_lanes = to_lanes
            ret = np.resize(x, expect_lanes)

    ret_vdtype = to_dtype.with_lanes(expect_lanes)
    return PyVar(_saturate_if_needed(ret, to_dtype, saturate), ret_vdtype)


@register_ir_api
def i(x):
    """Converts the given expression to the signed type with the same bits and lanes.

    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    x : tir.PrimExpr
        The expression that needs to be cast to signed type.

    Returns
    -------
    ret : tir.PrimExpr
        The result expression with the signed type.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32".

    Examples
    --------
    .. code-block:: python

        a0_i32 = S.i(a_ptr_u32[0])
        a0_i16 = S.i(a_ptr_u16[0] + vector_b)
        a0_i8 = S.i(a_ptr_u8[0] - 10)

    See Also
    --------
    - :doc:`../../language_basics/types`
    """
    assert isinstance(x, tir.PrimExpr), "The 1st arg expect a variable."
    x_dtype = x.dtype
    assert x_dtype.is_integer, f'The data type of arg must be integer, but got: "{x_dtype}".'
    return cast(x, x_dtype.with_int())


@register_ir_api
def _py_i(x):
    return x if x.dtype.is_int else PyVar(x.value, x.dtype.with_int())


@register_ir_api
def u(x):  # pylint: disable=invalid-name
    """Converts the given expression to the unsigned type with the same bits and lanes.

    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    x : tir.PrimExpr
        The expression that needs to be cast to unsigned type.

    Returns
    -------
    ret : tir.PrimExpr
        The result expression with the unsigned type.

    Supported DType
    ---------------
        "int8/16/32", "uint8/16/32".

    Examples
    --------
    .. code-block:: python

        a0_u32 = S.u(a_ptr_i32[0])
        a0_u16 = S.u(a_ptr_i16[0] + vector_b)
        a0_u8 = S.u(a_ptr_i8[0] - 10)

    See Also
    --------
    - :doc:`../../language_basics/types`
    """
    assert isinstance(x, tir.PrimExpr), "The 1st arg expect a variable."
    x_dtype = x.dtype
    assert x_dtype.is_integer, f'The data type of arg must be integer, but got: "{x_dtype}".'
    return cast(x, x_dtype.with_uint())


@register_ir_api
def _py_u(x):
    return x if x.dtype.is_uint else PyVar(x.value, x.dtype.with_uint())


def _gen_dtype_builder_impl(dtype):
    def _wrapper(init_value):
        """Create a new variable with the specified type and initial value.

        Parameters
        ----------
        init_value : Union[tir.PrimExpr, Literal["inf", "-inf", "nan"], int, float, list, tuple]
            The initial value of the new variable.

        Returns
        -------
        ret : tir.PrimExpr
            The new created variable.
        """
        return cast(init_value, dtype)

    _wrapper.__name__ = dtype
    try:
        _wrapper.type_ann_func = getattr(T, dtype)
    except AttributeError:
        _wrapper.type_ann_func = lambda: tir.Var(dtype, dtype)

    return _wrapper


def _gen_dtype_python_impl(dtype):
    def _wrapper(init_value):
        return _py_cast(init_value, dtype)

    _wrapper.__name__ = f"_py_{dtype}"
    return _wrapper


for _alias, _dtype in _ALIAS2DTYPE.items():
    globals()[_alias] = globals()[_dtype] = register_ir_api(_gen_dtype_builder_impl(_dtype))
    register_ir_api(_gen_dtype_python_impl(_dtype))


def _gen_size_dtype_builder_impl(dtype):
    def _wrapper(init_value):
        """Create a new variable that represent a vector index size whose value
        is always greater or equal to zero.

        Parameters
        ----------
        init_value : Union[tir.PrimExpr, int, list, tuple]
            The initial value of the new variable.

        Returns
        -------
        ret : tir.SizeVar
        The new created size variable.
        """
        init_value = cast(init_value, dtype)
        # Use a meaningless let expression to pass the necessary information to the
        # parser, so it can know this is a explicit variable definition statement.
        return tir.Let(tir.SizeVar("", dtype), init_value, tir.StringImm("define_size_var"))

    _wrapper.__name__ = f"size_{dtype}"
    _wrapper.type_ann_func = lambda: tir.SizeVar("", dtype)
    return _wrapper


def _gen_size_dtype_python_impl(dtype):
    def _wrapper(init_value):
        return _py_cast(init_value, dtype)

    _wrapper.__name__ = f"_py_size_{dtype}"
    return _wrapper


_SIZE_ALIAS2DTYPE = {f"size_{k}": f"size_{v}" for k, v in _ALIAS2DTYPE.items() if k[0] == "i"}
for _alias, _name in _SIZE_ALIAS2DTYPE.items():
    _dtype = _name[5:]
    globals()[_alias] = globals()[_name] = register_ir_api(_gen_size_dtype_builder_impl(_dtype))
    register_ir_api(_gen_size_dtype_python_impl(_dtype))


def _gen_bool_builder_impl(dtype):
    def _wrapper(*_):
        assert False, "The function only can be used in type annotation."

    _wrapper.__name__ = dtype
    _wrapper.type_ann_func = lambda: tir.Var("", dtype)
    return _wrapper


def _gen_empty_python_impl(name):
    def _wrapper():
        pass

    _wrapper.__name__ = f"_py_{name}"
    return _wrapper


_BOOL_DTYPES = ("bool", "boolx8", "boolx16", "boolx32")
for _dtype in _BOOL_DTYPES:
    globals()[_dtype] = register_ir_api(_gen_bool_builder_impl(_dtype))
    register_ir_api(_gen_empty_python_impl(_dtype))


@register_ir_api
def reinterpret(x, dtype):
    """Reinterprets the given expression to the specific data type.

    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    x : Union[Pointer, PrimExpr, int, float]
        The expression or value that need to be reinterpreted.

    dtype : Union[str, DataType]
        The target data type.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Examples
    --------
    .. code-block:: python

        a0_u32 = S.reinterpret(a_ptr_i32[0], "u32")
        a0_i16 = S.reinterpret(a_ptr_u16[0] + vector_b, "i16")
        a0_u8 = S.reinterpret(a_ptr_i8[0] - 10, "u8")
        a0_u32 = S.reinterpret(1.2345, "u32")
        a0_fp16 = S.reinterpret(S.i16(1.2345), "fp16")
        a0_fp32x8 = S.reinterpret(S.u32x8(123), "fp32x8")

    See Also
    --------
    - :doc:`../../language_basics/types`
    """
    x = tir.const(x) if isinstance(x, (int, float)) else x
    is_to_ptr = False
    scope = "global"
    if isinstance(dtype, str):
        is_to_ptr = "*" in dtype
        dtype = dtype.replace("*", "").strip()
        dtype_item = dtype.split(" ")
        assert len(dtype_item) in (1, 2), "Invalid dtype string."
        if len(dtype_item) == 2:
            scope, dtype = dtype_item
            scope = "local" if scope == "private" else scope
    dtype = resolve_dtype_alias(dtype)
    if is_to_ptr:
        msg = f'The scope expect one of ("private", "global", "lsram", "shared"), but got {scope}.'
        assert scope in ("local", "global", "lsram", "shared"), msg
        assert not isinstance(x, tir.Pointer), 'Please use "as_ptr" if convert ptr to ptr.'
        return tir.Pointer(dtype, scope, base=tir.call_intrin("handle", "tir.reinterpret", x))
    if not isinstance(x, tir.Pointer):
        msg = f'The total bits of the 1st arg expect {dtype.total_bits}, but got: "{x.dtype}".'
        assert x.dtype.total_bits == dtype.total_bits, msg
        assert_not_flexible_width_vector(x.dtype)
    else:
        msg = f'"Reinterpret pointer to number expect to_dtype is 32-bit, bug got: "{dtype}".'
        assert dtype.is_integer_scalar and dtype.bits == 32, msg
    return tir.call_intrin(dtype, "tir.reinterpret", x)


@register_ir_api
def _py_reinterpret(x, dtype):
    is_to_ptr = False
    scope = "global"
    if isinstance(dtype, str):
        is_to_ptr = "*" in dtype
        dtype = dtype.replace("*", "").strip()
        dtype_item = dtype.split(" ")
        if len(dtype_item) == 2:
            scope, dtype = dtype_item
            scope = "local" if scope == "private" else scope
    dtype = resolve_dtype_alias(dtype)
    if is_to_ptr:
        assert isinstance(x, PyPointer)
        return PyPointer(dtype, scope, x.u8_np_arr, x.u8_offset)
    if isinstance(x, PyPointer):
        return x.as_ptr("uint8")
    x_np = np.asarray(x, get_dtype(x).element_of)
    return PyVar(x_np.view(dtype.element_of), dtype)


@register_ir_api
def vxtl(x):
    """Extends the low half elements of ``x`` to double size. Sign-extends if signed dtype else
    zero-extends.

    .. code-block::

         x(i16x16): 1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16

        out = S.vxtl(x)
        out(i32x8): 1     2     3     4     5     6     7     8

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
        "int8/16", "uint8/16".

    Examples
    --------
    .. code-block:: python

        vc = S.vxtl(vx)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vsxtl, __vuxtl
    """
    assert is_vector(x), "The 1st arg expect a vector."
    x_vdtype = x.dtype
    assert_neither_flexible_nor_multiple_width_vector(x_vdtype)
    assert x_vdtype.bits != 32, "Can't extend to 64bit."
    assert x_vdtype.is_integer, "Only support integer instruction."
    ret_vdtype = double_elem_width(x_vdtype)
    return tir.call_extern(ret_vdtype, "vxtl", x)


@register_ir_api
def _py_vxtl(x):
    ret_vdtype = double_elem_width(x.dtype)
    return PyVar(x[: x.dtype.lanes // 2], ret_vdtype)


@register_ir_api
def vxth(x):
    """Extends the high half elements of ``x`` to double size. Sign-extends if signed dtype else
    zero-extends.

    .. code-block::

         x(i16x16): 1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16

        out = S.vxth(x)
        out(i32x8): 9     10    11    12    13    14    15    16

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
        "int8/16", "uint8/16".

    Examples
    --------
    .. code-block:: python

        vc = S.vxth(vx)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vsxth, __vuxth
    """
    assert is_vector(x), "The 1st arg expect a vector."
    x_vdtype = x.dtype
    assert_neither_flexible_nor_multiple_width_vector(x_vdtype)
    assert x_vdtype.bits != 32, "Can't extend to 64bit."
    assert x_vdtype.is_integer, "Only support integer instruction."
    ret_vdtype = double_elem_width(x_vdtype)
    return tir.call_extern(ret_vdtype, "vxth", x)


@register_ir_api
def _py_vxth(x):
    ret_vdtype = double_elem_width(x.dtype)
    return PyVar(x[x.dtype.lanes // 2 :], ret_vdtype)


__all__ = (
    tuple(_ALIAS2DTYPE.keys())
    + tuple(_ALIAS2DTYPE.values())
    + tuple(_SIZE_ALIAS2DTYPE.keys())
    + tuple(_SIZE_ALIAS2DTYPE.values())
    + _BOOL_DTYPES
    + (
        "cast",
        "i",
        "u",
        "reinterpret",
        "vxtl",
        "vxth",
    )
)
