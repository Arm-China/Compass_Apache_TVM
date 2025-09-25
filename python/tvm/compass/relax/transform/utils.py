# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Common Compass transform utilities."""
from tvm import relax, ir


def is_call(call, name):
    """Is a call node with op as given name."""
    name = name if name.startswith("relax.") else "relax." + name
    return isinstance(call, relax.Call) and call.op == ir.Op.get(name)


def is_compass_func(func):
    """Is a compass function"""
    return "Codegen" in func.attrs and func.attrs["Codegen"] == "compass"


def is_cps_composite_func(func):
    """Is a composite function"""
    return "Composite" in func.attrs and func.attrs["Composite"].startswith("compass")


def get_inverse_axes(axes):
    """Get inverse axes for permute_dims."""
    axes_dict = {axis: i for i, axis in enumerate(axes)}
    ordered_d = dict(sorted(axes_dict.items()))
    inverse_axes = list(ordered_d.values())
    return inverse_axes
