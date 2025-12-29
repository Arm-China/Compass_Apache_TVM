# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""The IR part of Zhouyi Compass script APIs."""
import os
from . import axis
from .ir import *
from .arithmetic import *  # pylint: disable=redefined-builtin
from .bitwise import *
from .compare import *  # pylint: disable=redefined-builtin
from .conversion import *
from .math import *  # pylint: disable=redefined-builtin
from .memory import *
from .miscellaneous import *
from .permutation import *
from .synchronization import *
from .base import ir_api_register_check


_exclude_names = (
    "os",
    "ir_api_register_check",
)
_exclude_names += tuple(x[:-3] for x in os.listdir(os.path.dirname(__file__)) if x.endswith(".py"))

__all__ = tuple(x for x in dir() if x[0] != "_" and x not in _exclude_names) + ("axis",)
ir_api_register_check()
