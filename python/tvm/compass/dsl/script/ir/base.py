# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The basic part of IR APIs."""
import inspect
import functools
from collections import defaultdict
import numpy as np
from tvm import tir, DataType
from ..pysim import PySimInfo, PyVar


_IR_API_NAME_2_IMPLEMENTS = defaultdict(lambda: [None, None])


def register_ir_api(func):
    """Register the given function as a IR API implementation.

    Each IR API has 2(i.e., IRBuilder and Python) implementations, the 1st one
    is used when the Zhouyi Compass DSL program is compiling, and the 2nd one is
    used when the Zhouyi Compass DSL program is running directly by the Python
    interpreter.
    The function name of the Python version must contain that of the IRBuilder
    version and with a extra prefix "_py_", e.g., "vadd" vs. "_py_vadd".
    """
    if func.__module__.startswith("tvm.compass.dsl.script."):
        name = func.__name__
    else:
        # Some IR API functions of TVM Script are created through generator, so
        # their function name is same, e.g., that of T.int32 and T.uint32 is
        # "tvm.script.xxx.func_gen.<locals>.func".
        caller_code = inspect.getframeinfo(inspect.currentframe().f_back).code_context[0]
        name = caller_code.split("=")[0].strip()

    if name[0] == "_":
        assert name[:4] == "_py_", "Decorate wrong function."
        name = name[4:]
        assert _IR_API_NAME_2_IMPLEMENTS[name][1] is None, f"{name} Override happened."
        _IR_API_NAME_2_IMPLEMENTS[name][1] = func
        return func

    assert _IR_API_NAME_2_IMPLEMENTS[name][0] is None, f"{name} Override happened."
    _IR_API_NAME_2_IMPLEMENTS[name][0] = func

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        if PySimInfo.current is None:
            return _IR_API_NAME_2_IMPLEMENTS[name][0](*args, **kwargs)

        # Execute the PySim version of the IR API.
        tld = PySimInfo.current.thread_local_data
        old_value, tld.is_in_ir_api = tld.is_in_ir_api, True
        ret = _IR_API_NAME_2_IMPLEMENTS[name][1](*args, **kwargs)
        tld.is_in_ir_api = old_value
        return ret

    return _wrapper


def ir_api_register_check():
    """Check if the IR API implementations are registered correctly."""
    for name, impls in _IR_API_NAME_2_IMPLEMENTS.items():
        assert impls[0] is not None, f'IR API "{name}" missing the IRBuilder implementation.'
        assert impls[1] is not None, f'IR API "{name}" missing the Python implementation.'


def _convert_mask_str(mask_str):
    # check and delete '_'
    err_msg = "The first and last character of mask string can not be '_'."
    assert len(mask_str) > 1 and mask_str[0] != "_" and mask_str[-1] != "_", err_msg
    assert "__" not in mask_str, "Can not use continuous '_'."
    mask_list = list(mask_str)
    for i in range(len(mask_list)):
        if mask_list[i] == "_":
            assert mask_list[i - 1] in ("T", "F"), "Can not write a digit number before '_'."
            if mask_list[i + 1] in ("T", "F"):
                mask_list[i] = "1"  # 3T_F --> 3T1F
            else:
                del mask_list[i]
    mask_str = "".join(mask_list)

    assert len(mask_str) > 1 and mask_str[-1] in ("T", "F"), (
        "The mask string need at least contains 2 characters and must end with "
        f'"T" or "F", but got: "{mask_str}".'
    )

    pattern_idx = None
    repeat_num_idx = None
    repeat_num = 1
    full_mask_str = ""

    idx = 0
    while idx < len(mask_str):
        cur_char = mask_str[idx]
        if cur_char.isdigit():
            if repeat_num_idx is None:  # Indicate it's the 1st character of the new repeat number.
                repeat_num_idx = idx
                if pattern_idx is not None:
                    full_mask_str += mask_str[pattern_idx:idx] * repeat_num
                    pattern_idx = None

            idx += 1  # Just move forward when it's the body of the new repeat number.
            continue

        assert cur_char in ("T", "F"), (
            'The mask string only can contain uppercase character "T", "F" and'
            f'decimal digit number, but got: "{cur_char}".'
        )

        if pattern_idx is None:  # Indicate it's the 1st character of the new pattern.
            pattern_idx = idx
            if repeat_num_idx is not None:
                repeat_num = int(mask_str[repeat_num_idx:idx])
                msg = 'The repeat number in mask string must >= 1, but got: "{repeat_num}".'
                assert repeat_num >= 1, msg
                repeat_num_idx = None

        idx += 1  # Just move forward when it's the body of the new pattern.

    full_mask_str += mask_str[pattern_idx:] * repeat_num
    return [{"T": True, "F": False}[x] for x in full_mask_str]


def _ir_builder_canonicalize_mask(mask, lanes):
    if isinstance(mask, tir.PrimExpr):
        mask_dtype = mask.dtype
        msg = f'The arg "mask" expect a boolean vector, but got: "{mask_dtype}".'
        assert mask_dtype.is_bool_vector, msg
        msg = f'The length of arg "mask" expect "{lanes}", but got: "{mask_dtype.lanes}".'
        assert mask_dtype.lanes == lanes, msg
        return mask

    if isinstance(mask, np.ndarray):
        msg = f'The arg "mask" expect a boolean vector, but got: "{mask.dtype}".'
        assert DataType(mask.dtype).is_bool, msg
        assert mask.ndim == 1, f'The arg "mask" expect a 1D boolean vector, but got: "{mask.ndim}".'
        msg = f'The length of arg "mask" expect <= "{lanes}", but got: "{len(mask)}".'
        assert len(mask) <= lanes, msg
        mask = [bool(x) for x in mask]

    if mask is None:
        ret = [True] * lanes
    else:
        msg = f'The arg "mask" expect a tuple or list, but got: "{type(mask)}".'
        assert isinstance(mask, (tuple, list)), msg
        msg = 'All elements of arg "mask" expect "True" or "False".'
        assert all(isinstance(x, bool) for x in mask), msg
        ret = list(mask) + [False] * (lanes - len(mask))

    return tir.const_pred(ret)


def canonicalize_mask(mask, lanes):
    """Canonicalize all supported constant mask forms (None, list, tuple, numpy.ndarray, str) to the
    one consistent form "tir.const_pred"."""
    mask = _convert_mask_str(mask) if isinstance(mask, str) else mask

    if PySimInfo.current is None:
        return _ir_builder_canonicalize_mask(mask, lanes)

    # Implementation of PySim.
    if isinstance(mask, PyVar):
        return mask.value

    if isinstance(mask, np.ndarray):
        mask = [bool(x) for x in mask]

    if mask is None:
        ret = [True] * lanes
    else:
        ret = list(mask) + [False] * (lanes - len(mask))

    return np.array(ret)
