# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Zhouyi Compass script APIs."""
from .parser import prim_func, macro
from .ir import *  # pylint: disable=redefined-builtin


def __getattr__(name):
    # pylint: disable=import-outside-toplevel
    pattern = r"^(i|size_i|u|fp|int|size_int|uint|float|bool)(8x|16x|32x|x)([1-9]\d*)$"

    import re

    match = re.match(pattern, name)
    if not match:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    type_code = match.group(1)
    bits = match.group(2)
    lanes = match.group(3)

    if (type_code in ("fp", "float") and bits == "8x") or (type_code != "bool" and bits == "x"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .ir.base import register_ir_api as ria

    if type_code == "bool":
        from .ir.conversion import _gen_bool_builder_impl, _gen_empty_python_impl

        globals()[name] = ria(_gen_bool_builder_impl(name))
        ria(_gen_empty_python_impl(name))
        return globals()[name]

    is_size_type = False
    if name.startswith("size_"):
        is_size_type = True
        name = name.replace("size_", "")

    type_mapping = {"i": "int", "u": "uint", "fp": "float"}
    reverse_type_mapping = {v: k for k, v in type_mapping.items()}
    # Indicate name is a _alias(e.g. i8x30, u8x30, fp32x9), need to expand to _dtype.
    if len(type_code) in (1, 2):
        _alias, _dtype = name, "".join([type_mapping[type_code], bits, lanes])
    else:
        _alias, _dtype = "".join([reverse_type_mapping[type_code], bits, lanes]), name

    from .ir.conversion import _gen_dtype_builder_impl, _gen_dtype_python_impl

    globals()[_alias] = globals()[_dtype] = ria(_gen_dtype_builder_impl(_dtype))
    ria(_gen_dtype_python_impl(_dtype))

    from .ir.conversion import _gen_size_dtype_builder_impl, _gen_size_dtype_python_impl

    globals()[f"size_{_alias}"] = globals()[f"size_{_dtype}"] = ria(
        _gen_size_dtype_builder_impl(_dtype)
    )
    ria(_gen_size_dtype_python_impl(_dtype))

    if is_size_type:
        return globals()[f"size_{_alias}"]
    return globals()[_alias]
