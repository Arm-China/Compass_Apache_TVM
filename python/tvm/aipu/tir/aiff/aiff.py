# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The main part of AIFF APIs."""
import copy
import numpy as np
from tvm import target as tgt
from .descriptor import DESC_CHAIN_EOF, DESC_CHAIN_NOE, CtrlDescChain, ParamDescChain, ActDescChain
from .descriptor import DescChainArray
from .utils import align8


def _gen_ctrl_desc_item(addr2reg, reg_addrs, begin_idx, end_idx):
    ret = []
    count = end_idx - begin_idx
    ret.append(count << 16 | reg_addrs[begin_idx])
    ret += [addr2reg[x].all_fields for x in reg_addrs[begin_idx:end_idx]]
    ret += [0] * (align8(count + 1) - (count + 1))
    return ret


class RegisterConfigBase:
    """The base class of the combination of all AIFF function units."""

    def __init__(self):
        self.unb = None
        self.wrb = None
        self.mtp = None
        self.itp = None
        self.ptp = None

    def get_register(self, addr):
        for x in (self.unb, self.wrb, self.mtp, self.itp, self.ptp):
            ret = x.get_register(addr)
            if ret is not None:
                return ret
        return None

    def _get_ctrl_addr2reg(self):
        ret = {}
        for x in (self.mtp, self.itp, self.ptp, self.wrb, self.unb):
            ret.update(x.get_ctrl_addr2reg(self))
        return ret

    def _get_param_regs(self):
        ret = []
        for x in (self.unb, self.wrb, self.mtp, self.itp, self.ptp):
            ret += x.get_param_regs()
        return ret

    def _get_act_regs(self):
        ret = []
        for x in (self.unb, self.wrb, self.mtp, self.itp, self.ptp):
            ret += x.get_act_regs()
        return ret

    def gen_descriptor(self):
        """Serialize the current register configuration to descriptor format, include "control",
        "parameter" and "activation" parts."""
        ctrl_addr2reg = self._get_ctrl_addr2reg()
        ctrl_reg_addrs = list(ctrl_addr2reg.keys())
        ctrl_desc = [0] * 8

        i = 1
        begin_idx = 0
        while i < len(ctrl_reg_addrs):
            if ctrl_reg_addrs[i] != ctrl_reg_addrs[i - 1] + 1:
                ctrl_desc += _gen_ctrl_desc_item(ctrl_addr2reg, ctrl_reg_addrs, begin_idx, i)
                begin_idx = i
            i += 1

        ctrl_desc += _gen_ctrl_desc_item(ctrl_addr2reg, ctrl_reg_addrs, begin_idx, i)

        # Parameter and activation descriptor parts.
        param_regs = self._get_param_regs()
        param_desc = [0] * align8(max(x.index_in_desc for x in param_regs) + 1)
        for x in param_regs:
            param_desc[x.index_in_desc] = x.all_fields

        act_regs = self._get_act_regs()
        act_desc = [0] * align8(max(x.index_in_desc for x in act_regs) + 1)
        for x in act_regs:
            act_desc[x.index_in_desc] = x.all_fields

        return ctrl_desc, param_desc, act_desc


def _new_register_config(aipu_info):
    from . import x2, x3  # pylint: disable=import-outside-toplevel

    return x2.RegisterConfig() if aipu_info.is_x2 else x3.RegisterConfig()


def _deserialize_register_config(aipu_info, desc, begin_idx, end_idx):
    ret = _new_register_config(aipu_info)
    cur_idx = begin_idx

    while cur_idx < end_idx:
        begin_addr = desc[cur_idx] & 0x0000_07FF  # Occupy 11 bits [10:0].
        count = (desc[cur_idx] & 0x003F_FFFF) >> 16  # Occupy 5 bits [21:16].

        for i in range(count):
            reg = ret.get_register(begin_addr + i)
            msg = f"The register with address '{begin_addr + i}' could not found"
            msg += f" in the AIFF function units for target '{aipu_info.name}'."
            assert reg is not None, msg
            reg.all_fields = desc[cur_idx + 1 + i]

        cur_idx += align8(count + 1)  # 256-bit aligned, so round up to multiple of 8.

    return ret


def _create_register_configs(aipu_info, desc):
    if desc is None:
        return [_new_register_config(aipu_info)]

    desc = np.fromfile(desc, dtype="uint32") if isinstance(desc, str) else desc
    msg = f'The descriptor expect a uint32 NumPy ndarray, but got: "{type(desc)}".'
    assert isinstance(desc, np.ndarray) and desc.dtype == "uint32", msg
    desc = desc.reshape(-1)

    ret = []
    i = 0
    eof_or_noe = None

    while eof_or_noe != DESC_CHAIN_EOF:
        eof_or_noe, _, _, cur_length_in_u32x8, loop_cfg, *_ = desc[i : i + 8]
        assert eof_or_noe in (DESC_CHAIN_NOE, DESC_CHAIN_EOF)
        cur_length_in_u32 = cur_length_in_u32x8 * 8

        reg_cfg = _deserialize_register_config(aipu_info, desc, i + 8, i + cur_length_in_u32)
        ret.append(reg_cfg)
        loop_remain_cnt = loop_cfg & 0x0000_FFFF
        ret += [copy.deepcopy(reg_cfg) for x in range(loop_remain_cnt)]
        i += cur_length_in_u32

    return ret


class Aiff:
    """The user interface of AIFF configuration.

    Each instance represents once start interaction between TEC and AIFF, after the start
    interaction, AIFF can execute multiple times autonomously, as for how many time it will be run
    depend on the count of register configurations in the instance. In other words, each instance
    will generate one descriptor chain.

    Each register configuration represents once AIFF execution, in other words, each register
    configuration will generate one node in the descriptor chain.
    """

    def __init__(self, target="X2_1204", descriptor=None):
        self._aipu_info = tgt.AipuInfo.get(target)
        msg = f'The Compass DSL AIFF does not support the target "{self._aipu_info.name}".'
        assert not self._aipu_info.is_x1, msg
        self.reg_cfgs = _create_register_configs(self._aipu_info, descriptor)

    @property
    def sys(self):
        assert not self._aipu_info.is_x3, 'There is not "sys" in X3 AIFF.'
        return self.reg_cfgs[-1].sys

    @property
    def unb(self):
        return self.reg_cfgs[-1].unb

    @property
    def wrb(self):
        return self.reg_cfgs[-1].wrb

    @property
    def mtp(self):
        return self.reg_cfgs[-1].mtp

    @property
    def itp(self):
        return self.reg_cfgs[-1].itp

    @property
    def ptp(self):
        return self.reg_cfgs[-1].ptp

    def add_new_register_config(self, idx=None, copy_idx=None):
        if copy_idx is not None:
            assert isinstance(copy_idx, int) and copy_idx < len(self.reg_cfgs)
            new_reg_cfg = copy.deepcopy(self.reg_cfgs[copy_idx])
        else:
            new_reg_cfg = _new_register_config(self._aipu_info)

        if idx is not None:
            assert isinstance(idx, int) and idx <= len(self.reg_cfgs)
        self.reg_cfgs.insert(len(self.reg_cfgs) if idx is None else idx, new_reg_cfg)

    def gen_descriptor(self):
        """Generate the descriptor chain according to the value of all register configurations.

        Each register configuration will "control", "parameter" and "activation" 3 descriptors, the
        "control" descriptors of all register configurations make a descriptor chain, so does
        "parameter" and "activation" descriptors.

        Returns
        -------
        ret : DescChainArray
            The result array of all descriptor chains, for this API, there must be 3 chains and the
            order is guaranteed to be "control", "parameter", and then "activation".
        """
        ctrl_descs, param_descs, act_descs = [], [], []
        for reg_cfg in self.reg_cfgs:
            ctrl_desc, param_desc, act_desc = reg_cfg.gen_descriptor()
            ctrl_descs.append(ctrl_desc)
            param_descs.append(param_desc)
            act_descs.append(act_desc)

        # Do the descriptor loop optimized for ctrl/param chain.
        ctrl_del_id, param_del_id = [], []
        i = 0
        while i < len(ctrl_descs):
            j = i + 1
            while j < len(ctrl_descs) and ctrl_descs[i] == ctrl_descs[j]:
                j += 1
            repeat_times = j - i - 1
            if repeat_times > 0:
                ctrl_descs[i][4] = repeat_times
                ctrl_del_id.insert(0, (i, j))
                if all(param_descs[i] == x for x in param_descs[i + 1 : j]):
                    param_del_id.insert(0, (i, j))
                else:
                    ctrl_descs[i][4] |= 1 << 16
            i = j
        for del_id in ctrl_del_id:
            ctrl_descs = ctrl_descs[: del_id[0] + 1] + ctrl_descs[del_id[1] :]
        for del_id in param_del_id:
            param_descs = param_descs[: del_id[0] + 1] + param_descs[del_id[1] :]

        act_chain = ActDescChain(act_descs, self._aipu_info)
        return DescChainArray((CtrlDescChain(ctrl_descs), ParamDescChain(param_descs), act_chain))
