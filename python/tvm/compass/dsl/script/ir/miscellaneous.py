# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""The miscellaneous part of IR APIs."""
import threading
import numpy as np
from tvm import tir, DataType
from tvm.script import tir as T
from ..pysim import PyVar, PySimInfo, pysim_run_sim
from .base import register_ir_api, canonicalize_mask
from .utils import is_scalar, is_integer_scalar, is_float_scalar, assert_lanes_valid
from .utils import assert_neither_flexible_nor_multiple_width_vector


@register_ir_api
def const_mask(x):
    """Creates a constant mask through boolean array or a special formatted string.

    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    Parameters
    ----------
    x : Union[Tuple[bool], List[bool], numpy.ndarray[bool], str]
        The concrete boolean values of the constant mask. String can be used to represent a boolean
        array. The mask string only can contain uppercase character ``T``, ``F`` and decimal digit
        numbers, ``T`` means True, and ``F`` means False. To represent multiple repeated boolean
        patterns, just add numbers before the characters, e.g., ``4T`` means ``TTTT``, and ``4TF``
        means ``TFTFTFTF``.

    Returns
    -------
    ret : PrimExpr
        The const mask result.

    Examples:
    ---------
    .. code-block:: python

        mask = S.const_mask([False] * 4 + [True] * 4)
        mask = S.const_mask("FFFFTTTT")
        mask = S.const_mask("4F4T")
        mask = S.const_mask("FFFF4T")

        mask = S.const_mask([False, True, False, True, True, False, True, False])
        mask = S.const_mask("2FT2TF")

        c = S.vadd(a, b, mask=mask)

    See Also
    --------
    - :doc:`../../language_basics/mask`
    """
    msg = "The 1st arg expect a boolean list or mask string."
    assert isinstance(x, (tuple, list, np.ndarray, str)), msg
    x_len = len(x)
    assert x_len > 1, f'The length of the 1st arg expect > 1, but got: "{x_len}".'
    return canonicalize_mask(x, x_len)


@register_ir_api
def _py_const_mask(x):
    return PyVar(canonicalize_mask(x, len(x)), DataType(f"boolx{len(x)}"))


@register_ir_api
def tail_mask(n, lanes):
    """Creates a mask that the lowest ``n`` items are True, and others are False.

    - The feature Flexible Width Vector is supported.
    - The feature Multiple Width Vector is supported.

    .. code-block::

        out = S.tail_mask(2, 8)
        out: T  T  F  F  F  F  F  F

    Parameters
    ----------
    n : Union[PrimExpr, int]
        The lowest item count will be set to True.

    lanes : int
        The total item count of the mask.

    Returns
    -------
    ret : PrimExpr
        The masked result.

    Examples:
    ---------
    .. code-block:: python

        if tail != 0:
            mask = S.tail_mask(tail, 16)

    See Also
    --------
    - :doc:`../../language_basics/mask`
    """
    assert_lanes_valid(lanes)
    return tir.low_true_pred(n, lanes)


@register_ir_api
def _py_tail_mask(n, lanes):
    n, lanes = int(n), int(lanes)
    assert 0 <= n <= lanes, f'The arg "n" expect in range [0, {lanes}], but got: "{n}".'
    return PyVar([True] * n + [False] * (lanes - n), DataType(f"boolx{lanes}"))


@register_ir_api
def get_local_size():
    """Returns the number of local work-items specified in the dimension identified by dimension
    index. For the Zhouyi NPU, returns the TEC number.

    Returns
    -------
    ret : int
        The TEC number.

    Examples
    --------
    .. code-block:: python

        tec_num = S.get_local_size()

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: get_local_size
    """
    return tir.call_extern("int32", "get_local_size", 0)


@register_ir_api
def _py_get_local_size():
    return PyVar(PySimInfo.current.local_size, DataType("int32"))


@register_ir_api
def get_local_id():
    """Returns the unique local work-item ID value for the dimension identified by dimension index.
    For the Zhouyi NPU, returns the TEC ID from 0 to TEC_NUM-1.

    Returns
    -------
    ret : int
        The TEC ID.

    Examples
    --------
    .. code-block:: python

        tid = S.get_local_id()

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: get_local_id
    """
    return tir.call_extern("int32", "get_local_id", 0)


@register_ir_api
def _py_get_local_id():
    return PyVar(PySimInfo.current.thread_local_data.id, DataType("int32"))


@register_ir_api
def tec_range(start, stop=None):
    """The explicit TEC parallel For statement.

    Parameters
    ----------
    start : Union[PrimExpr, int]
        The start of For_range.

    stop : Union[PrimExpr, int]
        The stop of For_range.

    Returns
    -------
    ret : frame.ForFrame
        The thread-binding For statement.

    Note
    ----
    If you pass only 1 argument to ``S.tec_range``, it will automatically set ``start = 0, stop =
    args[0]``.

    Examples
    --------
    .. code-block:: python

        tec_num = S.get_local_size()
        for tid in S.tec_range(tec_num):
            xxx  # tid = 0,1,2,3

        for tid in S.tec_range(1,3):
            xxx  # tid = 1,2

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: get_local_id, get_local_size
    """
    return T.thread_binding(start, stop, thread="threadIdx.x")


@register_ir_api
def _py_tec_range(start, stop=None):
    local_id = PySimInfo.current.thread_local_data.id
    if stop is None:
        stop = start
        start = 0
    # For the scenario that local_id = 0 and S.tec_range(1, 4), here should
    # return a empty container so the for loop will not be executed.
    return set([local_id]).intersection(range(start, stop))


@register_ir_api
def perf_tick_begin(cid):
    """Used for the profiler, begins recording the current tick count.

    Parameters
    ----------
    cid: int
        The custom ID, unique identifier of code fragment for performance monitoring.

    Examples
    --------
    .. code-block:: python

        S.perf_tick_begin(0)
        for i in range(10):
            c[i : i + 8] = a[i : i + 8] + b[i : i + 8]
        S.perf_tick_end(0)

        S.perf_tick_begin(1)
        for i in range(10):
            xxx
            xxx
        S.perf_tick_end(1)

    See Also
    --------
    - :doc:`../../how_to_guides/how_to_use_profiler`
    """
    return tir.call_extern("void", "__perf_record_tick_begin", cid)


@register_ir_api
def _py_perf_tick_begin(cid):  # pylint: disable=unused-argument
    pass  # There isn't corresponding concept in PySim, so needn't to be implemented.


@register_ir_api
def perf_tick_end(cid):
    """Used for the profiler, ends recording the current tick count.

    Parameters
    ----------
    cid: int
        The custom ID, unique identifier of code fragment for performance monitoring.

    Examples
    --------
    .. code-block:: python

        S.perf_tick_begin(0)
        for i in range(10):
            c[i : i + 8] = a[i : i + 8] + b[i : i + 8]
        S.perf_tick_end(0)

        S.perf_tick_begin(1)
        for i in range(10):
            xxx
            xxx
        S.perf_tick_end(1)

    See Also
    --------
    - :doc:`../../how_to_guides/how_to_use_profiler`
    """
    return tir.call_extern("void", "__perf_record_tick_end", cid)


@register_ir_api
def _py_perf_tick_end(cid):  # pylint: disable=unused-argument
    pass  # There isn't corresponding concept in PySim, so needn't to be implemented.


@register_ir_api
def aiff(ctrl_desc, param_desc, act_desc):
    """Do AIFF computation synchronously.

    Parameters
    ----------
    ctrl_desc : Pointer
        The pointer that store the control descriptor address.

    param_desc : Pointer
        The pointer that store the parameter descriptor address.

    act_desc : Pointer
        The pointer that store the activation descriptor address.

    Examples
    --------
    .. code-block:: python

        S.aiff(ctrl_desc, param_desc, act_desc)
    """
    msg = 'The arg "{}" expect a pointer.'
    assert isinstance(ctrl_desc, (tir.Buffer, tir.Pointer)), msg.format("ctrl_desc")
    assert isinstance(param_desc, (tir.Buffer, tir.Pointer)), msg.format("param_desc")
    assert isinstance(act_desc, (tir.Buffer, tir.Pointer)), msg.format("act_desc")
    ctrl_desc = ctrl_desc.addr_of(0) if isinstance(ctrl_desc, tir.Buffer) else ctrl_desc
    param_desc = param_desc.addr_of(0) if isinstance(param_desc, tir.Buffer) else param_desc
    act_desc = act_desc.addr_of(0) if isinstance(act_desc, tir.Buffer) else act_desc

    return tir.call_extern("void", "AIFF", ctrl_desc, param_desc, act_desc, 0)


@register_ir_api
def _py_aiff(ctrl_desc, param_desc, act_desc):
    ctrl = ctrl_desc.get_current_desc_chain()
    param = param_desc.get_current_desc_chain()
    act = act_desc.get_current_desc_chain()
    descs = [("ctrl", ctrl), ("param", param), ("act", act)]
    pysim_run_sim("AIFF(ctrl, param, act, 0);", descs=descs)


@register_ir_api
def async_aiff(ctrl_desc, param_desc, act_desc, event):
    """Do AIFF computation asynchronously.

    Parameters
    ----------
    ctrl_desc : Pointer
        The pointer that store the control descriptor address.

    param_desc : Pointer
        The pointer that store the parameter descriptor address.

    act_desc : Pointer
        The pointer that store the activation descriptor address.

    event : PrimExpr
        The event need to be triggered when the AIFF computation is completed. Note if the event is
        using by others, the AIFF hardware will be blocked until the event is triggered by others,
        then the AIFF computation will start. The API ``S.wait_events`` can be used to wait the AIFF
        computation operation to finish.

    Examples
    --------
    .. code-block:: python

        ev = S.alloc_event(1)
        S.async_aiff(ctrl_desc, param_desc, act_desc, ev)
        vc = va + vb
        S.wait_event(ev)
    """
    msg = 'The arg "{}" expect a pointer.'
    assert isinstance(ctrl_desc, (tir.Buffer, tir.Pointer)), msg.format("ctrl_desc")
    assert isinstance(param_desc, (tir.Buffer, tir.Pointer)), msg.format("param_desc")
    assert isinstance(act_desc, (tir.Buffer, tir.Pointer)), msg.format("act_desc")
    ctrl_desc = ctrl_desc.addr_of(0) if isinstance(ctrl_desc, tir.Buffer) else ctrl_desc
    param_desc = param_desc.addr_of(0) if isinstance(param_desc, tir.Buffer) else param_desc
    act_desc = act_desc.addr_of(0) if isinstance(act_desc, tir.Buffer) else act_desc

    return tir.call_extern("void", "ASYNC_AIFF", ctrl_desc, param_desc, act_desc, event, 0)


@register_ir_api
def _py_async_aiff(ctrl_desc, param_desc, act_desc, event):
    ctrl = ctrl_desc.get_current_desc_chain()
    param = param_desc.get_current_desc_chain()
    act = act_desc.get_current_desc_chain()
    descs = [("ctrl", ctrl), ("param", param), ("act", act)]

    idx = event.increase_producer()

    def _run():
        event.occupy_as_producer(idx)
        pysim_run_sim("AIFF(ctrl, param, act, 0);", descs=descs)
        event.trigger_as_producer(idx)

    threading.Thread(target=_run, name="async_aiff").start()


def _get_spcfr(fmt, start_idx):
    specifiers = ("%d", "%i", "%u", "%o", "%x", "%X", "%f", "%c", "%s", "%p", "%%")
    specifiers += ("%v32hhx", "%v16hx", "%v8hlx", "%v16hf", "%v8hlf")
    for spcfr in specifiers:
        if fmt.startswith(spcfr, start_idx):
            return spcfr
    raise ValueError(f'Unsupported specifier "{fmt[start_idx:]}".')


def _visit_spcfr(fmt, callback_func):
    cur_idx = 0
    while cur_idx < len(fmt):
        if fmt[cur_idx] != "%":
            callback_func(0, fmt[cur_idx])
            cur_idx += 1
        else:
            spcfr = _get_spcfr(fmt, cur_idx)
            callback_func(1, spcfr)
            cur_idx += len(spcfr)


def _check_arg_type(fmt, args):
    arg_cnt = len(args)
    cur_arg_idx = 0

    def _callback(invoke_pos, spcfr):
        nonlocal cur_arg_idx
        if invoke_pos != 1 or spcfr == "%%":
            return

        if cur_arg_idx >= arg_cnt:
            cur_arg_idx += 1
            return

        arg = args[cur_arg_idx]
        if spcfr == "%c":
            assert is_integer_scalar(arg), f'Specifier "{spcfr}" only accept integer scalar value.'
        elif spcfr == "%s":
            msg = f'Specifier "{spcfr}" only accept string value.'
            assert isinstance(arg, str) or arg.dtype == "handle", msg
        elif spcfr in ("%d", "%i", "%u", "%o"):
            assert is_integer_scalar(arg), f'Specifier "{spcfr}" only accept integer scalar value.'
        elif spcfr in ("%x", "%X"):
            assert is_scalar(arg), f'Specifier "{spcfr}" only accept scalar value.'
        elif spcfr == "%f":
            assert is_float_scalar(arg), f'Specifier "{spcfr}" only accept float scalar value.'
        elif spcfr in ("%v32hhx", "%v16hx", "%v8hlx"):
            msg = f'Specifier "{spcfr}" only accept 256-bit vector value.'
            assert isinstance(arg, tir.PrimExpr) and arg.dtype.total_bytes == 32, msg
        elif spcfr in ("%v16hf", "%v8hlf"):
            msg = f'Specifier "{spcfr}" only accept 256-bit float vector value.'
            assert isinstance(arg, tir.PrimExpr), msg
            assert arg.dtype.is_float and arg.dtype.total_bytes == 32, msg
        cur_arg_idx += 1

    _visit_spcfr(fmt, _callback)

    msg = f'The specifier count "{cur_arg_idx}" not equal to the argument count "{arg_cnt}".'
    assert cur_arg_idx == arg_cnt, msg


@register_ir_api
def printf(fmt, *args):
    """The printf built-in function, same as C printf.

    Parameters
    ----------
    fmt : str
        The format string, such as ``x=%d``.

    \\*args : Optional[Union[Tuple[PrimExpr, int, float]]]
        The items to be printed.

    Note
    ----
    In extremely rare cases, the results of DSL programs differ between using and not using
    ``printf``,  one possible reason is that using ``printf`` may make the optimization of the
    underlying OpenCL compiler conservative.

    Examples
    --------
    .. code-block:: python

        # Scalar
        S.printf("tec_num = %d\\n", S.get_local_size())
        S.printf("tec_num = %d, tec_id = %d\\n", tec_num, S.get_local_id())

        # Integer vector
        S.printf("%v32hhx, %v32hhx\\n", va_i8x32, vb_u8x32)
        S.printf("%v16hx, %v16hx\\n", S.i16x16(i16_max), S.u16x16(u16_max))
        S.printf("%v8hlx, %v8hlx\\n", va_i32x8, vb_u32x8)

        # Floating vector
        S.printf("%v8hlf, %v8hlf\\n", S.fp32x8(1.25), va_fp32x8)
        S.printf("%v16hf, %v16hf\\n", S.fp16x16(1.2345678), va_fp16x16)

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: printf
    """
    assert isinstance(fmt, str), 'The arg "fmt" expect a string.'
    _check_arg_type(fmt, args)
    args = tuple(tir.const(x) if isinstance(x, (int, float)) else x for x in args)
    return tir.call_extern("int32", "printf", fmt, *args)


def _format_arg(spcfr, arg):
    if isinstance(arg, PyVar):
        arg = arg.value.reshape(-1)[0] if arg.dtype.is_scalar else arg.value

    if spcfr == "%p":
        return "%p"  # Meaningless for PySim, just leave it there.
    if spcfr == "%c":
        return f"{arg:c}"
    if spcfr == "%s":
        return arg
    if spcfr in ("%d", "%i"):
        return str(np.int32(arg))
    if spcfr == "%u":
        return str(np.uint32(arg))
    if spcfr == "%o":
        return f"0{np.uint32(arg):o}"
    if spcfr in ("%x", "%X"):
        v = np.uint32(arg) if isinstance(arg, (int, np.integer)) else np.float32(arg).view("uint32")
        return f"0x{v:x}" if spcfr == "%x" else f"0x{v:X}"

    if spcfr in ("%v32hhx", "%v16hx", "%v8hlx"):
        bit_cnt = 256 // (8 if spcfr == "%v8hlx" else int(spcfr[2:4]))
        return ",".join(f"0x{v:x}" for v in arg.view(f"uint{bit_cnt}"))

    assert spcfr in ("%f", "%v16hf", "%v8hlf")
    arg = (arg,) if not isinstance(arg, np.ndarray) else arg
    # Compass always keep 6 digits following the decimal point, and it won't do rounding, e.g.
    # "0.1234567" will be printed as "0.123456" instead of "0.123457", so here we use this method to
    # align with it.
    return ",".join(f"{np.float32(x):.12f}"[:-6] for x in arg)


@register_ir_api
def _py_printf(fmt, *args):
    filled_fmt = ""
    cur_arg_idx = 0

    def _callback(invoke_pos, x):
        nonlocal filled_fmt, cur_arg_idx
        if invoke_pos == 0:
            filled_fmt += x
        elif invoke_pos == 1:
            if x == "%%":
                filled_fmt += "%"
            else:
                filled_fmt += _format_arg(x, args[cur_arg_idx])
                cur_arg_idx += 1

    _visit_spcfr(fmt, _callback)
    print(filled_fmt, end="")
    return 0


def asm(template, outputs=None, inputs=None, clobbers=None, qualifiers=None):
    """Insert assembly instructions in Compass DSL source code.

    Parameters
    ----------
    template : str
        The literal string that consist of assembly code.

    outputs : Optional[Dict[str, Tuple[str, PrimExpr]]]
        The output dictionary which key is the symbolic name in template and value is a two element
        tuple. The first element of the tuple is the constraint string, and the second one is the
        variable in Compass DSL code that need replace the corresponding symbolic name in template.

    inputs : Optional[dict]
        The input dictionary which key is the symbolic name in template and value is a two element
        tuple. The first element of the tuple is the constraint string, and the second one is the
        variable in Compass DSL code that need replace the corresponding symbolic name in template.

    clobbers : Optional[Union[Tuple[str], List[str]]
        The registers or other values that are changed by the assembler template, beyond those
        listed in the output dictionary.

    qualifiers : Optional[str]
        The qualifier for Compass OpenCL compiler, valid choices: (``None``, ``"inline"``,
        ``"volatile"``).

    Examples
    --------
    .. code-block:: python

        S.asm(
            "{add t0.b, %[inp].b, 1;\n\t}\n\t"
            "{add %[out].b, %[inp].b, t0.b;}",
            outputs={"out": ("=&t", y)},
            inputs={"inp": ("t", x)},
            clobbers=["t0"],
            qualifiers="volatile",
        )

    See Also
    --------
    - Zhouyi Compass OpenCL Programming Guide: Inline assembly, __asm__
    """
    assert isinstance(template, str), 'The arg "template" expect a string.'
    outputs = {} if outputs is None else outputs
    inputs = {} if inputs is None else inputs
    clobbers = tuple() if clobbers is None else clobbers
    msg = 'The arg "qualifiers" expect one of (None, "inline", "volatile").'
    assert qualifiers in (None, "inline", "volatile"), msg
    qualifiers = "" if qualifiers is None else f" {qualifiers} "

    args = [tir.precodegen(qualifiers), template, ": "]
    # Add output operands.
    for i, (name, (constraint, expr)) in enumerate(outputs.items()):
        assert_neither_flexible_nor_multiple_width_vector(expr.dtype)
        args += [tir.precodegen(f'[{name}] "{constraint}"('), expr]
        args.append(tir.precodegen(")" if i == (len(outputs) - 1) else "), "))

    # Add input operands.
    args.append(": ")
    for i, (name, (constraint, expr)) in enumerate(inputs.items()):
        assert_neither_flexible_nor_multiple_width_vector(expr.dtype)
        args += [tir.precodegen(f'[{name}] "{constraint}"('), expr]
        args.append(tir.precodegen(")" if i == (len(inputs) - 1) else "), "))

    # Add clobbers.
    args.append(": ")
    args.append(tir.precodegen(", ".join(f'"{x}"' for x in clobbers)))
    return tir.call_extern("void", "inline_asm", *args)


def _py_asm(template, outputs=None, inputs=None, clobbers=None, qualifiers=None):
    raise NotImplementedError("PySim does not support inline assembly.")


__all__ = (
    "const_mask",
    "tail_mask",
    "get_local_size",
    "get_local_id",
    "tec_range",
    "perf_tick_begin",
    "perf_tick_end",
    "aiff",
    "async_aiff",
    "printf",
    "asm",
)
