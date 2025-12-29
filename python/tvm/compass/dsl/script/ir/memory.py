# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
# pylint: disable=redefined-outer-name
"""The memory part of IR APIs."""
import threading
import numpy as np
from tvm import tir, script, DataType
from tvm.script import tir as T
from ...utils import hw_native_vdtype, HW_NATIVE_STORAGE_DTYPES, HW_NATIVE_VDTYPES
from ...utils import resolve_dtype_alias, VALID_PTR_ELEMENT_DTYPES
from ..pysim import PyVar, PyBuffer, PyEvent, PySimInfo, random_pause
from .base import register_ir_api, canonicalize_mask
from .utils import is_vector, is_scalar_const, is_integer_scalar, within_range, assert_lanes_valid


_valid_ptr_scopes = ("global", "global.1", "global.2", "global.3")
_valid_ptr_scopes += ("lsram", "shared", "private", "constant")


@register_ir_api
def ptr(dtype, scope="private"):
    """Annotates a function parameter as a pointer.

    Parameters
    ----------
    dtype : Union[str, DataType]
        The data type of the data that the pointer points to.

        - Scalar dtype:``int8``, ``uint8``, ``int16``, ``uint6``, ``int32``, ``uint32``,
          ``float16``, ``float32``, ``bfloat16``, ``void``
        - Vector dtype:``int8x32``, ``uint8x32``, ``int16x16``, ``uint16x16``, ``int32x8``,
          ``uint32x8``, ``float16x16``, ``float32x8``, ``bfloat16x16``

    scope : Optional[str]
        The memory space of the data that the pointer points to. The valid choices are listed
        below.

        - **global:** Represent the global DDR space of Address Space Extension region ID (ASID) 0.
        - **global.1:** Represent the global DDR space of ASID 1.
        - **global.2:** Represent the global DDR space of ASID 2.
        - **global.3:** Represent the global DDR space of ASID 3.
        - **private:** Represent the stack space of each TEC.
        - **lsram:** Represent the local SRAM space of each TEC.
        - **shared:** Represent the shared SRAM space between all TECs in the same core.
        - **constant:** Represent the global constant DDR space.

    Returns
    -------
    ret : Pointer
        The pointer instance.

    Examples
    --------
    .. code-block:: python

        @S.prim_func
        def func(a: S.ptr("i8", "global"), b: S.ptr("i8", "global"), n: S.i32):
            for i in range(n):
                b[i] = a[i]

    See Also
    --------
    - :doc:`../../how_to_guides/how_to_use_pointer`
    """
    assert dtype not in (None, ""), 'Please use "void" to explicitly represent the void pointer.'
    dtype = resolve_dtype_alias(dtype)
    msg = f'The scalar form of arg "dtype" expect one of {VALID_PTR_ELEMENT_DTYPES}, but got: '
    msg += f'"{dtype.element_of}".'
    assert dtype.element_of in VALID_PTR_ELEMENT_DTYPES, msg

    msg = f'The arg "scope" expect one of {_valid_ptr_scopes}, but got:"{scope}".'
    assert scope in _valid_ptr_scopes, msg
    scope = "local" if scope == "private" else scope
    return tir.Pointer(dtype, scope)


@register_ir_api
def _py_ptr(dtype=None, scope="private"):  # pylint: disable=unused-argument
    pass  # Only will be used in type annotation, so needn't to be implemented.


@register_ir_api
def match_buffer(pointer, shape):
    """Matches a pointer with the specified shape.

    Parameters
    ----------
    pointer : Pointer
        The data pointer to be matched.

    shape : Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral]
        The data shape to match.

    Returns
    -------
    ret : Buffer
        The matched buffer.

    Examples
    --------
    .. code-block:: python

        # 2D transpose demo b[j, i] = a[i, j]
        @S.prim_func
        def func(A: S.ptr("int8", "global"), B: S.ptr("int8", "global"), h: S.int32, w: S.int32):
            a = S.match_buffer(A, shape=(h, w))
            b = S.match_buffer(B, shape=(w, h))

            for i,j in S.grid(h, w):
                b[j, i] = a[i, j]
    """
    shape = (shape,) if not isinstance(shape, (list, tuple)) else shape
    assert all(is_integer_scalar(x) for x in shape)
    pointer.accessible_check(0)
    args = (pointer.base, shape, pointer.dtype, None, [], None, pointer.scope, -1, 0, "", None)
    return script.ir_builder.tir._ffi_api.MatchBuffer(*args)


@register_ir_api
def _py_match_buffer(pointer, shape):
    return PyBuffer(pointer.dtype, shape, pointer.scope, pointer.u8_np_arr, pointer.u8_offset)


_valid_alloc_dtypes = HW_NATIVE_STORAGE_DTYPES + HW_NATIVE_VDTYPES
_valid_alloc_scopes = ("lsram", "shared", "private")


@register_ir_api
def alloc_buffer(shape, dtype, scope="private"):
    """The buffer allocation function, returns the allocated buffer.

    Parameters
    ----------
    shape : Union[List[int], Tuple[int], int]
        The shape of buffer.

    dtype : Union[str, DataType]
        The data type of buffer elements. Can be scalar dtype or vector dtype.

    scope : Optional[str]
        The memory space in which the data is allocated. The valid choices are listed below.

        - **private:** Represent the stack space of each TEC.
        - **lsram:** Represent the local SRAM space of each TEC.
        - **shared:** Represent the shared SRAM space between all TECs in the same core.

    Returns
    -------
    ret : Buffer
        The allocated buffer.

    Examples
    --------
    .. code-block:: python

        lsram_a = S.alloc_buffer([32, 32], dtype, scope="lsram")
        S.dma_copy(lsram_a, ddr_a, 32 * 32)
        b = lsram_a[10, 5] + 2

    See Also
    --------
    - :doc:`../../how_to_guides/how_to_use_pointer`
    """
    shape = (shape,) if not isinstance(shape, (list, tuple)) else shape
    msg = "The element of 1st arg expect an integer scalar constant."
    assert all(isinstance(x, int) for x in shape), msg

    dtype = resolve_dtype_alias(dtype)
    msg = f'The arg "dtype" expect one of {_valid_alloc_dtypes}, but got: "{dtype}".'
    assert str(dtype) in _valid_alloc_dtypes, msg

    msg = f'The arg "scope" expect one of {_valid_alloc_scopes}, but got: "{scope}".'
    assert scope in _valid_alloc_scopes, msg
    scope = "local" if scope == "private" else scope
    return T.alloc_buffer(shape=shape, dtype=dtype, scope=scope)


@register_ir_api
def _py_alloc_buffer(shape, dtype, scope="private"):
    shape = (shape,) if not isinstance(shape, (list, tuple)) else tuple(shape)
    dtype = resolve_dtype_alias(dtype)
    scope = "local" if scope == "private" else scope

    size_in_byte = np.prod(shape + (dtype.lanes,) if dtype.is_vector else shape) * dtype.bytes
    ret = PyBuffer(dtype, shape, scope, np.empty(shape=(size_in_byte,), dtype="uint8"))
    if scope != "shared":
        return ret

    py_sim_info = PySimInfo.current
    assert py_sim_info.is_multi_thread, "PySim can't be run in single thread for this DSL program."

    if py_sim_info.thread_local_data.id == 0:
        py_sim_info.cur_shared_buffer = ret
    # This barrier to wait thread 0 put its buffer to global cur_shared_buffer
    # otherwise other threads would fetch wrong buffer.
    py_sim_info.barrier.wait()
    ret = py_sim_info.cur_shared_buffer
    # Second barrier to avoid thread 0 run very fast than other threads and put
    # another alloc shared buffer to cur_shared_buffer before other threads
    # fetch the current one.
    py_sim_info.barrier.wait()
    random_pause()
    return ret


@register_ir_api
def alloc(shape, dtype, scope="private"):
    """Allocates buffer with shape, dtype, scope, and returns the pointer.

    Parameters
    ----------
    shape : Union[List[int], Tuple[int], int]
        The shape of buffer.

    dtype : Union[str, DataType]
        The data type of buffer elements. Can be scalar dtype or vector dtype.

    scope : Optional[str]
        The memory space in which the data is allocated. The valid choices are listed below.

        - **private:** Represent the stack space of each TEC.
        - **lsram:** Represent the local SRAM space of each TEC.
        - **shared:** Represent the shared SRAM space between all TECs in the same core.

    Returns
    -------
    ret : Pointer
        The allocated buffer pointer.


    Examples
    --------
    .. code-block:: python

        lsram_a = S.alloc(1024, "int8", scope="lsram")
        S.dma_copy(lsram_a, ddr_a, 1024)

        lsram_a = S.alloc(256, "float16x16", scope="shared")
        lsram_a = S.alloc((256,), "int8", scope="lsram")
        lsram_a = S.alloc([1024], "int8", scope="lsram")

    See Also
    --------
    - :doc:`../../how_to_guides/how_to_use_pointer`
    """
    return alloc_buffer(shape, dtype, scope).addr_of(0)


@register_ir_api
def _py_alloc(shape, dtype, scope="private"):
    return _py_alloc_buffer(shape, dtype, scope).addr_of(0)


@register_ir_api
def alloc_const(shape, dtype, data):
    """Allocates constant data.

    Parameters
    ----------
    shape : Union[List[PrimExpr], Tuple[PrimExpr]]
        The shape of const buffer.

    dtype : Union[str, DataType]
        The data type of const buffer elements.

    data : np.array
        The data of const buffer.

    Returns
    -------
    ret : Buffer
        The allocated buffer.

    Examples
    --------
    .. code-block:: python

        # The "lut_data" can be created in pure Python environment during compile time.
        lut_data = np.array(list(range(512)),dtype="float16")

        @S.prim_func
        def func(inp: S.ptr("fp16", "global"), out: S.ptr("fp16", "global")):
            lut = S.alloc_const((512,), "float16", lut_data)
            ...

    """
    return np.asarray(data, dtype).reshape(shape)


@register_ir_api
def _py_alloc_const(shape, dtype, data):
    u8_np_arr = np.array(data, dtype).view("uint8").reshape((-1,))
    return PyBuffer(DataType(dtype), shape, "constant", u8_np_arr)


@register_ir_api
def vload(ptr, mask=None, lanes=None, stride=1):
    """Load a vector from contiguous or strided memory addresses.

    - The inactive elements of result vector are set to zero.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    ptr : Pointer
        The pointer that store the base memory address.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    lanes : Optional[int]
        The lanes of result vector dtype. If omitted, will be automatically determined based on
        the type of input address.

    stride : Optional[Union[PrimExpr, int]]
        The stride of each element. Will take one element every so many stride intervals.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Examples
    --------
    .. code-block:: python

        va = S.vload(ptr_a)
        va = S.vload(ptr_a, mask="3T5F")
        va = S.vload(ptr_a, mask=S.tail_mask(n, 8))
        va = S.vload(ptr_a, mask="1T31F", lanes=32)
        va = S.vload(ptr_a, mask="16T16F", lanes=32, stride=4)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vload, __vload_stride
    """
    ptr = ptr.addr_of(0) if isinstance(ptr, tir.Buffer) else ptr
    src_dtype = ptr.dtype
    ret_dtype = hw_native_vdtype(src_dtype) if src_dtype.is_scalar else src_dtype
    if lanes is not None:
        assert_lanes_valid(lanes)
        ret_dtype = src_dtype.with_lanes(lanes)

    assert is_integer_scalar(stride), 'The arg "stride" expect an integer scalar.'

    mask = canonicalize_mask(mask, ret_dtype.lanes)
    return tir.call_extern(ret_dtype, "vload", ptr, stride, mask)


@register_ir_api
def _py_vload(ptr, mask=None, lanes=None, stride=1):
    ptr = ptr.addr_of(0) if isinstance(ptr, PyBuffer) else ptr
    src_dtype = ptr.dtype
    ret_vdtype = hw_native_vdtype(src_dtype) if src_dtype.is_scalar else src_dtype
    if lanes is not None:
        ret_vdtype = src_dtype.with_lanes(lanes)

    assert stride >= 1, f'The arg "stride" expect >= 1, but got: "{stride}".'

    mask = canonicalize_mask(mask, ret_vdtype.lanes)
    ret = PyVar.zeros(ret_vdtype)  # The inactive elements are 0.
    scalar_ptr = ptr.as_ptr(ret_vdtype.element_of)

    for i in range(ret_vdtype.lanes):
        if mask[i]:
            ret[i] = scalar_ptr[i * stride]
    return ret


@register_ir_api
def vstore(value, ptr, mask=None, stride=1):
    """Store active elements of ``value`` to contiguous or strided memory addresses.

    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    value : PrimExpr
        The vector that needs to be stored.

    ptr : Pointer
        The pointer that store the base memory address.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    stride : Optional[Union[PrimExpr, int]]
        The stride of each element. Will store one element every so many stride intervals.

    Examples
    --------
    .. code-block:: python

        S.vstore(va, ptr_a)
        S.vstore(va, ptr_a, mask="3T5F")
        S.vstore(va, ptr_a, mask=S.tail_mask(n, 8))
        S.vstore(va, ptr_a, mask="T7F", stride=4)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vstore, __vstore_stride
    """
    assert is_vector(value), 'The arg "value" expect a vector.'
    ptr = ptr.addr_of(0) if isinstance(ptr, tir.Buffer) else ptr
    assert is_integer_scalar(stride), 'The arg "stride" expect an integer scalar.'

    mask = canonicalize_mask(mask, value.dtype.lanes)
    return tir.call_extern("void", "vstore", value, ptr, stride, mask)


@register_ir_api
def _py_vstore(value, ptr, mask=None, stride=1):
    ptr = ptr.addr_of(0) if isinstance(ptr, PyBuffer) else ptr
    mask = canonicalize_mask(mask, value.dtype.lanes)
    assert stride >= 1, f'The arg "stride" expect >= 1, but got: "{stride}".'

    scalar_ptr = ptr.as_ptr(value.dtype.element_of)
    for i in range(value.dtype.lanes):
        if mask[i]:
            scalar_ptr[i * stride] = value[i]


@register_ir_api
def vload_gather(ptr, indices, mask=None):
    """Load a vector from discrete memory addresses. The addresses are calculated by
    ``ptr + indices * elem_size_in_byte``.

    - The inactive elements of result vector are set to zero.
    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    ptr : Pointer
        The pointer that store the base memory address.

    indices : PrimExpr
        The indices used to calculate the discrete memory addresses, it must be a 16-bit integer
        vector, its length decide the length of result vector.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    Returns
    -------
    ret : PrimExpr
        The result expression.

    Examples
    --------
    .. code-block:: python

        va = S.vload_gather(ptr_a, vb)
        va = S.vload_gather(ptr_a, vb, mask="T7F")
        va = S.vload_gather(ptr_a, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vload_gather
    """
    ptr = ptr.addr_of(0) if isinstance(ptr, tir.Buffer) else ptr
    assert is_vector(indices), 'The arg "indices" expect a vector.'
    indices_vdtype = indices.dtype
    msg = f'The arg "indices" expect a 16-bit integer vector, but got: "{indices_vdtype}".'
    assert indices_vdtype.is_integer and indices_vdtype.bits == 16, msg

    ret_vdtype = ptr.dtype.with_lanes(indices_vdtype.lanes)
    offsets = indices * tir.const(ptr.dtype.bytes, dtype=indices_vdtype)
    mask = canonicalize_mask(mask, ret_vdtype.lanes)
    return tir.call_extern(ret_vdtype, "vload_gather", ptr, offsets, mask)


@register_ir_api
def _py_vload_gather(ptr, indices, mask=None):
    ptr = ptr.addr_of(0) if isinstance(ptr, PyBuffer) else ptr
    ret_vdtype = ptr.dtype.with_lanes(indices.dtype.lanes)
    mask = canonicalize_mask(mask, ret_vdtype.lanes)

    ret = PyVar.zeros(ret_vdtype)
    scalar_ptr = ptr.as_ptr(ret_vdtype.element_of)

    for i in range(ret_vdtype.lanes):
        if mask[i]:
            ret[i] = scalar_ptr[indices[i]]
    return ret


@register_ir_api
def vstore_scatter(value, ptr, indices, mask=None):
    """Store active elements of ``value`` to discrete memory addresses. The addresses are calculated
    by ``ptr + indices * elem_size_in_byte``.

    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    value : PrimExpr
        The vector that need to be stored.

    ptr : Pointer
        The pointer that store the base memory address.

    indices : PrimExpr
        The indices used to calculate the discrete memory addresses, it must be a 16-bit integer
        vector, its length should be equal to that of ``value``.

    mask : Optional[Union[Tuple[bool], List[bool], numpy.ndarray[bool], str, PrimExpr]]
        The predication mask to indicate which elements of the vector are active for the operation.
        ``None`` means all elements are active.

    Examples
    --------
    .. code-block:: python

        S.vstore_scatter(va, ptr_a, vb)
        S.vstore_scatter(va, ptr_a, vb, mask="3T5F")
        S.vstore_scatter(va, ptr_a, vb, mask=S.tail_mask(n, 8))

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __vstore_scatter
    """
    ptr = ptr.addr_of(0) if isinstance(ptr, tir.Buffer) else ptr
    assert is_vector(value), 'The arg "value" expect a vector.'
    assert is_vector(indices), 'The arg "indices" expect a vector.'

    indices_vdtype = indices.dtype
    msg = f'The arg "indices" expect a 16-bit integer vector, but got: "{indices_vdtype}".'
    assert indices_vdtype.is_integer and indices_vdtype.bits == 16, msg

    value_vdtype = value.dtype
    msg = f'The lanes mismatch: "value": "{value_vdtype}" vs. "indices": "{indices_vdtype}".'
    assert value_vdtype.lanes == indices_vdtype.lanes, msg
    msg = f'The element type mismatch: "value": "{value_vdtype}" vs. "ptr": "{ptr.dtype}*".'
    assert value_vdtype.element_of == ptr.dtype.element_of, msg

    offsets = indices * tir.const(ptr.dtype.bytes, dtype=indices_vdtype)
    mask = canonicalize_mask(mask, value_vdtype.lanes)
    return tir.call_extern("void", "vstore_scatter", value, ptr, offsets, mask)


@register_ir_api
def _py_vstore_scatter(value, ptr, indices, mask=None):
    ptr = ptr.addr_of(0) if isinstance(ptr, PyBuffer) else ptr
    mask = canonicalize_mask(mask, value.dtype.lanes)

    scalar_ptr = ptr.as_ptr(value.dtype.element_of)
    for i in range(value.dtype.lanes):
        if mask[i]:
            scalar_ptr[indices[i]] = value[i]


@register_ir_api
def dma_copy(dst, src, width, src_stride=None, times=1, dst_stride=None):
    """Copy the specified number of elements from the source address to the destination address
    via DMA.

    Parameters
    ----------
    dst : Pointer
        The pointer that store the destination memory address.

    src : Pointer
        The pointer that store the source memory address.

    width : Union[PrimExpr, int]
        The number of data to be transfer inside one stride jump.

    src_stride : Optional[Union[PrimExpr, int]]
        The number of source data will be jump over for each stride jump. ``None`` means equal with
        the value of ``width``, i.e., load from the source memory address continuously.

    times : Optional[Union[PrimExpr, int]]
        The total times of the stride jump.

    dst_stride : Optional[Union[PrimExpr, int]]
        The number of destination data will be jump over for each stride jump. ``None`` means equal
        with the value of ``width``, i.e., store to the destination memory address continuously.

    Notes
    -----
    - The pointer type of ``src`` and ``dst`` must be same.
    - Only below scope combinations of ``src`` and ``dst`` are not supported.

      - The scope of ``src`` is ``lsram`` and the scope of ``dst`` is ``lsram`` or ``shared``.
      - The scope of ``src`` is ``shared`` and the scope of ``dst`` is ``lsram`` or ``shared``.

    Examples
    --------
    .. code-block:: python

        # The 1D scenario.
        S.dma_copy(ptr_a, ptr_b, 16)

        # The 2D scenario. Transfer all "@" in "@@@@@xxx@@@@@xxx@@@@@xxx@@@@@xxx" and store them
        # continuously in destination.
        S.dma_copy(ptr_a, ptr_b, width=5, src_stride=8, times=4)

    See Also
    --------
    - :doc:`../../how_to_guides/how_to_use_dma`
    """
    assert isinstance(dst, (tir.Buffer, tir.Pointer)), 'The arg "dst" expect a pointer.'
    assert isinstance(src, (tir.Buffer, tir.Pointer)), 'The arg "src" expect a pointer.'
    dst = dst.addr_of(0) if isinstance(dst, tir.Buffer) else dst
    src = src.addr_of(0) if isinstance(src, tir.Buffer) else src

    msg = f'The data type mismatch: "dst": "{dst.dtype}*" vs. "src": "{src.dtype}*".'
    assert dst.dtype == src.dtype, msg

    src_scope, dst_scope = src.scope, dst.scope
    msg = f'Currently does not support copy data from scope "{src_scope}" to scope "{dst_scope}".'
    assert src_scope != "constant" and dst_scope != "constant", msg
    msg = f'Does not support copy data from scope "{src_scope}" to scope "{dst_scope}".'
    assert not all(x in ("lsram", "shared") for x in (src_scope, dst_scope)), msg

    itemsize = src.dtype.bytes
    src_stride = width if src_stride is None else src_stride
    dst_stride = width if dst_stride is None else dst_stride

    return tir.call_extern(
        "void",
        "DmaDirect",
        dst,
        src,
        width * itemsize,
        src_stride * itemsize,
        width * times * itemsize,
        dst_stride * itemsize,
    )


_MAX_LSRAM_IN_BYTE = 32 * 1024
_MAX_SHARED_IN_BYTE = 256 * 1024
_UINT32_MAX = 2**32 - 1


@register_ir_api
def _py_dma_copy(dst, src, width, src_stride=None, times=1, dst_stride=None):
    dst = dst.addr_of(0) if isinstance(dst, PyBuffer) else dst
    src = src.addr_of(0) if isinstance(src, PyBuffer) else src
    src_stride = width if src_stride is None else src_stride
    dst_stride = width if dst_stride is None else dst_stride

    assert width >= 0, f'The arg "width" expect >= 0, but got: "{width}".'
    assert src_stride >= width, f'The arg "src_stride" expect >= {width}, but got: "{src_stride}".'
    assert dst_stride >= width, f'The arg "dst_stride" expect >= {width}, but got: "{dst_stride}".'

    itemsize = src.dtype.bytes
    max_lsram = _MAX_LSRAM_IN_BYTE // itemsize
    max_shared = _MAX_SHARED_IN_BYTE // itemsize
    max_count = _UINT32_MAX // itemsize
    total_count = width * times

    msg = f'The product of arg "width" and "times" expect <= {max_count}, but got: "{total_count}".'
    assert total_count <= max_count, msg

    if src.scope == "lsram" or dst.scope == "lsram":
        assert width <= max_lsram, f'The arg "width" expect <= {max_lsram}, but got: "{width}".'

    if src.scope in ("lsram", "shared"):
        max_limit = max_lsram if src.scope == "lsram" else max_shared
        msg = f'The arg "src_stride" expect <= {max_limit}, but got: "{src_stride}".'
        assert src_stride <= max_limit, msg
        msg = f'The product of arg "width" and "times" expect <= {max_limit}, '
        msg += f'but got: "{total_count}".'
        assert total_count <= max_limit, msg

    if dst.scope in ("lsram", "shared"):
        max_limit = max_lsram if dst.scope == "lsram" else max_shared
        msg = f'The arg "dst_stride" expect <= {max_limit}, but got: "{dst_stride}".'
        assert dst_stride <= max_limit, msg
        msg = f'The product of arg "width" and "times" expect <= {max_limit}, '
        msg += f'but got: "{total_count}".'
        assert total_count <= max_limit, msg

    for i in range(times):
        dst[i * dst_stride : i * dst_stride + width] = src[i * src_stride : i * src_stride + width]


@register_ir_api
def async_dma_copy(dst, src, width, src_stride=None, times=1, dst_stride=None, event=None):
    """Copy the specified number of elements from the source address to the destination address
    asynchronously via DMA.

    Once DMA's configuration is finished, this API will return and the behind code will be executed
    immediately, at the same time the data is transferring via DMA.

    Parameters
    ----------
    dst : Pointer
        The pointer that store the destination memory address.

    src : Pointer
        The pointer that store the source memory address.

    width : Union[PrimExpr, int]
        The number of data to be transfer inside one stride jump.

    src_stride : Optional[Union[PrimExpr, int]]
        The number of source data will be jump over for each stride jump. ``None`` means equal with
        the value of ``width``, i.e., load from the source memory address continuously.

    times : Optional[Union[PrimExpr, int]]
        The total times of the stride jump.

    dst_stride : Optional[Union[PrimExpr, int]]
        The number of destination data will be jump over for each stride jump. ``None`` means equal
        with the value of ``width``, i.e., store to the destination memory address continuously.

    event : PrimExpr
        The event need to be triggered when the entire data transmission is completed. Note if the
        event is using by others, the DMA hardware will be blocked until the event is triggered by
        others, then the data transmission will start. The API ``S.wait_events`` can be used to wait
        the DMA operation to finish.

    Notes
    -----
    - The pointer type of ``src`` and ``dst`` must be same.
    - Only below scope combinations of ``src`` and ``dst`` are not supported.

      - The scope of ``src`` is ``lsram`` and the scope of ``dst`` is ``lsram`` or ``shared``.
      - The scope of ``src`` is ``shared`` and the scope of ``dst`` is ``lsram`` or ``shared``.

    Examples
    --------
    .. code-block:: python

        ev0 = S.alloc_events(1)

        # The 1D scenario.
        S.async_dma_copy(ptr_a, ptr_b, 16, event=ev0)
        vc = va + vb
        S.wait_events(ev0)

        # The 2D scenario. Transfer all "@" in "@@@@@xxx@@@@@xxx@@@@@xxx@@@@@xxx" and store them
        # continuously in destination.
        S.async_dma_copy(ptr_a, ptr_b, width=5, src_stride=8, times=4, event=ev0)
        vc = va + vb
        S.wait_events(ev0)

    See Also
    --------
    - :doc:`../../how_to_guides/how_to_use_dma`
    """
    assert isinstance(dst, (tir.Buffer, tir.Pointer)), 'The arg "dst" expect a pointer.'
    assert isinstance(src, (tir.Buffer, tir.Pointer)), 'The arg "src" expect a pointer.'
    dst = dst.addr_of(0) if isinstance(dst, tir.Buffer) else dst
    src = src.addr_of(0) if isinstance(src, tir.Buffer) else src

    msg = f'The data type mismatch: "dst": "{dst.dtype}*" vs. "src": "{src.dtype}*".'
    assert dst.dtype == src.dtype, msg

    src_scope, dst_scope = src.scope, dst.scope
    msg = f'Currently does not support copy data from scope "{src_scope}" to scope "{dst_scope}".'
    assert src_scope != "constant" and dst_scope != "constant", msg
    msg = f'Does not support copy data from scope "{src_scope}" to scope "{dst_scope}".'
    assert not all(x in ("lsram", "shared") for x in (src_scope, dst_scope)), msg

    assert event is not None, 'The arg "event" must be provided.'

    itemsize = src.dtype.bytes
    src_stride = width if src_stride is None else src_stride
    dst_stride = width if dst_stride is None else dst_stride

    return tir.call_extern(
        "void",
        "AsyncDmaDirect",
        dst,
        src,
        width * itemsize,
        src_stride * itemsize,
        width * times * itemsize,
        dst_stride * itemsize,
        event,
    )


_DMA_MAX_WIDTH_IN_BYTE = 0xFFFF  # The corresponding register is 16-bit.
_DMA_MAX_STRIDE_IN_BYTE = 0xFFFFFF  # The corresponding register is 24-bit.
_DMA_MAX_TRANS_SIZE_IN_BYTE = 0xFFFFFF  # The corresponding register is 24-bit.


@register_ir_api
def _py_async_dma_copy(dst, src, width, src_stride=None, times=1, dst_stride=None, event=None):
    dst = dst.addr_of(0) if isinstance(dst, PyBuffer) else dst
    src = src.addr_of(0) if isinstance(src, PyBuffer) else src
    src_stride = width if src_stride is None else src_stride
    dst_stride = width if dst_stride is None else dst_stride

    assert width >= 0, f'The arg "width" expect >= 0, but got: "{width}".'
    assert src_stride >= width, f'The arg "src_stride" expect >= {width}, but got: "{src_stride}".'
    assert dst_stride >= width, f'The arg "dst_stride" expect >= {width}, but got: "{dst_stride}".'

    itemsize = src.dtype.bytes
    max_width = _DMA_MAX_WIDTH_IN_BYTE // itemsize
    max_stride = _DMA_MAX_STRIDE_IN_BYTE // itemsize
    max_count = _DMA_MAX_TRANS_SIZE_IN_BYTE // itemsize

    assert width <= max_width, f'The arg "width" expect <= "{max_width}", but got: "{width}".'
    msg = f'The arg "src_stride" expect <= {max_stride}, but got: "{src_stride}".'
    assert src_stride <= max_stride, msg
    msg = f'The arg "dst_stride" expect <= {max_stride}, but got: "{dst_stride}".'
    assert dst_stride <= max_stride, msg
    assert (
        width * times <= max_count
    ), f'The product of arg "width" and "times" expect <= {max_count}, but got: "{width * times}".'

    assert isinstance(event, PyEvent), 'The arg "event" expect object returned by "S.alloc_events".'
    assert not event.is_free, 'The arg "event" is a released event, can not be used anymore.'

    idx = event.increase_producer()

    def _run():
        event.occupy_as_producer(idx)
        PySimInfo.current.thread_local_data.is_in_ir_api = False

        for i in range(times):
            dst[i * dst_stride : i * dst_stride + width] = src[
                i * src_stride : i * src_stride + width
            ]

        event.trigger_as_producer(idx)

    threading.Thread(target=_run, name="async_dma_copy").start()


@register_ir_api
def dma_transpose2d(dst, src, row, col, dst_stride=None, src_stride=None):
    """Uses DMA to transpose for 2d data.

    Parameters
    ----------
    dst: Pointer
        The pointer that store the destination memory address.

    src: Pointer
        The pointer that store the source memory address.

    row: Union[PrimExpr, int]
        The row to be transposed of input 2d data [row, col].

    col: Union[PrimExpr, int]
        The col to be transposed of input 2d data [row, col].

    dst_stride: Optional[Union[PrimExpr, int]]
        The width_stride of output 2d data. Default dst_stride = row.

    src_stirde: Optional[Union[PrimExpr, int]]
        The width_stride of input 2d data. Default src_stride = col.

    Examples
    --------
    .. code-block:: python

        S.dma_transpose2d(dst, src, 8, 64)
    """
    assert isinstance(src, (tir.Buffer, tir.Pointer)), f"Unsupport addr type {type(src)}"
    assert isinstance(dst, (tir.Buffer, tir.Pointer)), f"Unsupport addr type {type(dst)}"
    src = src.addr_of(0) if isinstance(src, tir.Buffer) else src
    dst = dst.addr_of(0) if isinstance(dst, tir.Buffer) else dst

    dtype = src.dtype.with_lanes(1)
    assert dtype == dst.dtype.with_lanes(1), "src and dst should of same dtype"
    data_unit = int(dtype.bytes // 2)  # 0 for BYTE, 1 for HALF, 2 for WORD
    itemsize = dtype.bytes

    if src_stride is None:
        src_stride = col
    if dst_stride is None:
        dst_stride = row

    return tir.call_extern(
        "void", "DMA_Transpose2D", src, dst, col, row, src_stride, dst_stride, data_unit, itemsize
    )


@register_ir_api
def _py_dma_transpose2d(dst, src, row, col, dst_stride=None, src_stride=None):
    src = src.addr_of(0) if isinstance(src, PyBuffer) else src
    dst = dst.addr_of(0) if isinstance(dst, PyBuffer) else dst

    dtype = src.dtype.with_lanes(1)
    itemsize = dtype.bytes

    assert col >= 0, f"col={col} should >=0"
    assert row >= 0, f"row={row} should >=0"
    assert (
        col * itemsize <= 0xFFFFFF
    ), f"col*itemsize: {col*itemsize} exceed DMA_MAX_STRIDE 0xFFFFFF"
    assert (
        row * itemsize <= 0xFFFFFF
    ), f"row*itemsize{ {row*itemsize}} exceed DMA_MAX_STRIDE 0xFFFFFF"

    if src_stride is None:
        src_stride = col
    else:
        assert (
            col <= src_stride and col * itemsize <= 0xFFFF
        ), "please check col*itemsize<=0xFFFF and col<=src_stride"
    if dst_stride is None:
        dst_stride = row
    else:
        assert (
            row <= dst_stride and row * itemsize <= 0xFFFF
        ), "please check row*itemsize<=0xFFFF and row<=dst_stride"
    for i in range(row):
        for j in range(col):
            dst[j * dst_stride + i] = src[i * src_stride + j]


@register_ir_api
def dma_upsample(
    dst,
    src,
    h_scale,
    w_scale,
    c,
    w,
    src_c_stride=None,
    dst_c_stride=None,
    dst_w_stride=None,
):  # pylint: disable=invalid-name
    """DMA upsample data from source ``src`` to destination ``dst``. Supports two directions between
    ``src`` and ``dst``:
    1. global -> [lsram, shared].
    2. [lsram, shared] -> global.

    Note:
    1. Each call of ``dma_upsample`` does a surface on 2D input in WC layout physically, not a 3D
    input.
    2. If you want to upsample 3D input with ``h_scale``, ``w_scale`` for H, W dimensions
    respectively, you need to call H times ``dma_upsample``, where H is H dimension of input in HWC
    layout.

    Parameters
    ----------
    dst : Pointer
        The pointer that store the destination memory address.

    src : Pointer
        The pointer that store the source memory address.

    h_scale : Union[PrimExpr, int]
        The scale on h direction.

    w_scale : Union[PrimExpr, int]
        The scale on w direction.

    c : Union[PrimExpr, int]
        The c of each move on source.

    w : Union[PrimExpr, int]
        The w of each move on source.

    src_c_stride : Optional[Union[PrimExpr, int]]
        The c stride of each move on source. Default src_c_stride = c.

    dst_c_stride : Optional[Union[PrimExpr, int]]
        The c stride of each move on destination. Default dst_c_stride = c.

    dst_w_stride : Optional[Union[PrimExpr, int]]
        The w stride of each move on destination. Default dst_w_stride = w_scale * w.

    Examples
    --------
    .. code-block:: python

        # Case0: an easy use case
        S.dma_upsample(dst=ddr_ptr, src=sram_ptr, h_scale=2, w_scale=3, c=3, w=2)

        # Source 2D data in WC layout(w=2, c=3)
        # [
        #   [0 1 2],
        #   [3 4 5],
        # ]
        # Upsampled destination 3D data in HWC layout
        # [
        #   [
        #     [0 1 2],
        #     [0 1 2],
        #     [0 1 2],
        #
        #     [3 4 5],
        #     [3 4 5],
        #     [3 4 5],
        #   ],
        #   [
        #     [0 1 2],
        #     [0 1 2],
        #     [0 1 2],
        #
        #     [3 4 5],
        #     [3 4 5],
        #     [3 4 5],
        #   ],
        # ]

        # Case1: a comprehensive use case
        S.dma_upsample(dst=ddr_ptr, src=sram_ptr, h_scale=2, w_scale=3,
                     c=3, w=2, src_c_stride=4, dst_c_stride=5, dst_w_stride=7)

        # Source 2D data in WC layout(c=3, w=2, src_c_stride=4)
        # [
        #   [0 1 2 ?],
        #   [3 4 5 ?],
        # ]
        # Upsampled destination 3D data in HWC layout(dst_w_stride=7, dst_c_stride=5)
        # [
        #   [
        #     [0 1 2 ? ?],
        #     [0 1 2 ? ?],
        #     [0 1 2 ? ?],
        #
        #     [3 4 5 ? ?],
        #     [3 4 5 ? ?],
        #     [3 4 5 ? ?],
        #
        #     [? ? ? ? ?],
        #   ],
        #   [
        #     [0 1 2 ? ?],
        #     [0 1 2 ? ?],
        #     [0 1 2 ? ?],
        #
        #     [3 4 5 ? ?],
        #     [3 4 5 ? ?],
        #     [3 4 5 ? ?],
        #
        #     [? ? ? ? ?],
        #   ],
        # ]
    """
    assert isinstance(src, (tir.Buffer, tir.Pointer)), f"Unsupported src type {type(src)}"
    assert isinstance(dst, (tir.Buffer, tir.Pointer)), f"Unsupported dst type {type(dst)}"

    src = src.addr_of(0) if isinstance(src, tir.Buffer) else src
    dst = dst.addr_of(0) if isinstance(dst, tir.Buffer) else dst

    msg = f'Argument pointer type mismatch: 0-th: "{dst.dtype}" vs. 1-th: "{src.dtype}".'
    assert dst.dtype == src.dtype, msg
    assert (src.scope.startswith("global") and dst.scope in ("lsram", "shared")) or (
        src.scope in ("lsram", "shared") and dst.scope.startswith("global")
    ), "Only support upsample data between DDR(global) and SRAM(lsram, shared)."

    src_c_stride = c if src_c_stride is None else src_c_stride
    dst_c_stride = c if dst_c_stride is None else dst_c_stride
    dst_w_stride = w * w_scale if dst_w_stride is None else dst_w_stride

    itemsize = src.dtype.bytes
    return tir.call_extern(
        "void",
        "DmaUpsample",
        dst,
        src,
        h_scale,
        w_scale,
        c * itemsize,
        w,
        src_c_stride * itemsize,
        dst_c_stride * itemsize,
        dst_w_stride,
    )


@register_ir_api
def _py_dma_upsample(
    dst,
    src,
    h_scale,
    w_scale,
    c,
    w,
    src_c_stride=None,
    dst_c_stride=None,
    dst_w_stride=None,
):  # pylint: disable=invalid-name
    dst = dst.addr_of(0) if isinstance(dst, PyBuffer) else dst
    src = src.addr_of(0) if isinstance(src, PyBuffer) else src

    dst_c_stride = c if dst_c_stride is None else dst_c_stride
    dst_w_stride = w * w_scale if dst_w_stride is None else dst_w_stride
    src_c_stride = c if src_c_stride is None else src_c_stride

    assert src_c_stride >= c, f'"src_c_stride"({src_c_stride}) expect >= "c"({c}).'
    assert dst_c_stride >= c, f'"dst_c_stride"({dst_c_stride}) expect >= "c"({c}).'
    msg = f'"dst_w_stride"({dst_w_stride}) expect >= "w"({w}) * "w_scale"({w_scale}).'
    assert dst_w_stride >= w * w_scale, msg
    # The max w scale "255" is from register limited
    assert w_scale <= 255, f'"w_scale"({w_scale}) expect <= "max_w_scale"(255).'

    for widx in range(w):
        src_idx = widx * src_c_stride
        src_c = src[src_idx : src_idx + c]
        for hsidx in range(h_scale):
            for wsidx in range(w_scale):
                h_ofs = hsidx * dst_w_stride * dst_c_stride
                w_ofs = (widx * w_scale + wsidx) * dst_c_stride
                dst_idx = h_ofs + w_ofs
                dst[dst_idx : dst_idx + c] = src_c


@register_ir_api
def dma_memset(ptr, value, num):
    """Fills the first num elements of addr to the specific value.

    Parameters
    ----------
    ptr : Pointer
        The pointer that store the base memory address.

    value : Union[PrimExpr, int]
        The same dtype as addr dtype.
        The value to be set.

    num : Union[PrimExpr, int]
        The number of scalar elements that need to be set.

    Examples
    --------
    .. code-block:: python

        S.dma_memset(ptr_a, 0, 128)
    """
    assert isinstance(ptr, (tir.Buffer, tir.Pointer)), 'The arg "ptr" expect a pointer.'

    ptr = ptr.addr_of(0) if isinstance(ptr, tir.Buffer) else ptr
    dtype = ptr.dtype.with_lanes(1)

    data_unit = int(dtype.bytes // 2)  # 0 for BYTE, 1 for HALF, 2 for WORD
    trans_size = num * dtype.bytes

    # check value within range
    if is_scalar_const(value):
        assert within_range(value, dtype), f'value "{value}" out of range "{dtype}".'

    # imm value, convert to int at compile time
    # var value, reinterpret at runtime
    if dtype.is_float32:
        if is_scalar_const(value):
            value = np.float32(value).view("int32")
        else:
            value = tir.reinterpret("int32", value)
    if dtype.is_float16:
        if is_scalar_const(value):
            value = np.float16(value).view("int16")
        else:
            value = tir.reinterpret("int16", value)

    msg = f'Dose not support set value on scope "{ptr.scope}".'
    assert ptr.scope in ["global", "lsram", "shared", "local"], msg
    addr_base_flag = 0
    memset_func = "MemsetDDR"
    if ptr.scope == "lsram":
        addr_base_flag = 4
        memset_func = "MemsetSRAM"
    elif ptr.scope == "shared":
        addr_base_flag = 8
        memset_func = "MemsetSRAM"
    return tir.call_extern("void", memset_func, ptr, trans_size, value, data_unit, addr_base_flag)


@register_ir_api
def _py_dma_memset(ptr, value, num):
    ptr = ptr.addr_of(0) if isinstance(ptr, PyBuffer) else ptr
    dtype = ptr.dtype
    assert num >= 0, "dma_memset(ptr,value,num): num should >=0"
    num = num * dtype.lanes if dtype.is_vector else num

    ptr.as_ptr(dtype.element_of)[:num] = value


@register_ir_api
def flush_cache(invalidate=True):
    """Flushes the whole level 1 data cache by writing back to DDR.

    Parameters
    ----------
    invalidate : bool
        Whether invalidates the data or not.

    Examples
    --------
    .. code-block:: python

        S.flush_cache()
        S.flush_cache(invalidate=False)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: __flush_cache
    """
    return tir.call_extern("void", "__flush_cache", invalidate)


@register_ir_api
def _py_flush_cache(invalidate=True):  # pylint: disable=unused-argument
    pass  # There isn't corresponding concept in PySim, so needn't to be implemented.


__all__ = (
    "ptr",
    "match_buffer",
    "alloc",
    "alloc_buffer",
    "alloc_const",
    "vload",
    "vstore",
    "vload_gather",
    "vstore_scatter",
    "dma_copy",
    "async_dma_copy",
    "dma_transpose2d",
    "dma_upsample",
    "dma_memset",
    "flush_cache",
)
