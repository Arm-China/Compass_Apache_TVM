# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The common part of AIFF register."""
import re
import numpy as np
from ...logger import DEBUG


class FieldInfo:
    """Store the information of a register field."""

    def __init__(self, low_idx, high_idx):
        self.low_idx = low_idx
        self.high_idx = high_idx
        self.max_value = (1 << (self.high_idx - self.low_idx + 1)) - 1


_camel_to_snake_pattern = re.compile(r"(?<!^)(?=[A-Z])")
_UINT32_MAX = 2**32 - 1


class Register:
    """The base class of each register."""

    name = None
    field_infos = {}
    desc_type = "ctrl"

    def __init__(self, addr_in_u32, index_in_desc=None):
        self.addr = addr_in_u32

        if self.name is None:
            self.name = _camel_to_snake_pattern.sub("_", self.__class__.__name__).lower()

        self.index_in_desc = index_in_desc

    def __setattr__(self, name, value):
        if "index_in_desc" not in self.__dict__ or name == "all_fields":
            super().__setattr__(name, value)
            return

        info = self.field_infos.get(name)
        assert info is not None, f'The field "{name}" is not valid.'

        if self.desc_type in ("param", "act"):
            assert (isinstance(value, (int, np.integer)) and value == 0) or (
                isinstance(value, np.ndarray) and value.size != 0
            ), f'The field "{name}" expect a non-empty NumPy or 0, but got: "{repr(value)}".'
        else:
            msg = f'The field "{name}" expect in range [0, {info.max_value}], but got: '
            msg += f'"{value}".'
            assert isinstance(value, (int, np.integer)) and 0 <= value <= info.max_value, msg

        DEBUG(f"AIFF Register: 0x{self.addr:03X} {self.name}.{name} = {value}")

        super().__setattr__(name, value)

    @property
    def all_fields(self):
        if self.desc_type in ("param", "act"):
            return getattr(self, tuple(self.field_infos.keys())[0])

        ret = 0
        for name, info in self.field_infos.items():
            ret |= getattr(self, name) << info.low_idx

        return ret

    @all_fields.setter
    def all_fields(self, value):
        if self.desc_type in ("param", "act"):
            self.__setattr__(tuple(self.field_infos.keys())[0], value)
            return

        msg = f'The register expect in range [0, {_UINT32_MAX}], but got: "{value}".'
        assert isinstance(value, (int, np.integer)) and 0 <= value <= _UINT32_MAX, msg
        for name, info in self.field_infos.items():
            self.__dict__[name] = (value & ((1 << (info.high_idx + 1)) - 1)) >> info.low_idx
            DEBUG(f"AIFF Register: 0x{self.addr:03X} {self.name}.{name} = {self.__dict__[name]}")
