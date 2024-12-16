# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The common part of AIFF function unit."""
from .register import Register


class FunctionUnit:
    """The base class of each function unit."""

    def __init__(self):
        self._addr2reg = {x.addr: x for x in self.__dict__.values() if isinstance(x, Register)}

    def __setattr__(self, name, value):
        attr = self.__dict__.get(name)
        if isinstance(attr, Register):
            attr.all_fields = value
            return

        super().__setattr__(name, value)

    def get_register(self, addr):
        return self._addr2reg.get(addr)

    def get_ctrl_addr2reg(self, reg_cfg):  # pylint: disable=unused-argument
        addr2reg = {k: v for k, v in self._addr2reg.items() if v.desc_type == "ctrl"}
        return {k: addr2reg[k] for k in sorted(addr2reg.keys())}

    def get_param_regs(self):
        return [v for k, v in self._addr2reg.items() if v.desc_type == "param"]

    def get_act_regs(self):
        return [v for k, v in self._addr2reg.items() if v.desc_type == "act"]


class MultipleUnit(FunctionUnit):
    """The base class of the function unit that contain multiple sub-unit."""

    def __init__(self):
        super().__init__()
        self._units = tuple()

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        return self._units[idx]

    def get_register(self, addr):
        ret = self._addr2reg.get(addr)
        if ret is not None:
            return ret

        for unit in self._units:
            ret = unit.get_register(addr)
            if ret is not None:
                return ret
        return None

    def get_param_regs(self):
        ret = super().get_param_regs()
        for unit in self._units:
            ret += unit.get_param_regs()
        return ret

    def get_act_regs(self):
        ret = super().get_act_regs()
        for unit in self._units:
            ret += unit.get_act_regs()
        return ret
