# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Common AIPU Compass transform utilities."""
from tvm import relax, ir


def is_call(call, name):
    """Is a call node with op as given name."""
    name = name if name.startswith("relax.") else "relax." + name
    return isinstance(call, relax.Call) and call.op == ir.Op.get(name)
