# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""The descriptor part of AIFF APIs."""
import numpy as np


DESC_CHAIN_EOF = 0x81
DESC_CHAIN_NOE = 0x18


class _DescChain:
    def __init__(self, descs):
        self._descs = descs
        self._counts = tuple(len(x) for x in self._descs)
        self.count_of_u32 = sum(self._counts)

    def __iter__(self):
        return self._descs.__iter__()

    def get_item_as_flatten_list(self, idx):
        cur_idx = idx
        for count, desc in zip(self._counts, self._descs):
            if cur_idx < count:
                return desc[cur_idx]
            cur_idx -= count

        raise IndexError("The index is out of range.")

    def set_item_as_flatten_list(self, idx, value):
        cur_idx = idx
        for count, desc in zip(self._counts, self._descs):
            if cur_idx < count:
                desc[cur_idx] = value
                return
            cur_idx -= count

        raise IndexError("The index is out of range.")


class CtrlDescChain(_DescChain):
    """The "control" descriptor chain."""

    def __init__(self, descs):
        super().__init__(descs)

        for i, desc in enumerate(self._descs):
            desc[0] = DESC_CHAIN_EOF
            desc[3] = len(desc) // 8  # The length of the current descriptor, unit is 256-bit.

            if i != 0:
                self._descs[i - 1][0] = DESC_CHAIN_NOE
                self._descs[i - 1][2] = desc[3]


def _get_origin_np_arr(x):
    return x if x.base is None else x.base


def _find_io_const_np_arrs(descs, fcheck=None):
    np_arrs = []
    for desc in descs:
        for i, x in enumerate(desc):
            flag = True if fcheck is None else fcheck(i)
            if isinstance(x, np.ndarray) and flag:
                origin_np_arr = _get_origin_np_arr(x)
                if origin_np_arr not in np_arrs:
                    np_arrs.append(origin_np_arr)
    return np_arrs


class ParamDescChain(_DescChain):
    """The "parameter" descriptor chain."""

    @property
    def const_np_arrs(self):
        return _find_io_const_np_arrs(self._descs)


_OUTPUT_REGISTER_INDICES = {
    "X2": (16, 17, 40, 41, 42, 43),
    "X3P": (16, 17, 19, 64, 65),
    "X3S": (16, 17, 19, 64, 65),
}


class ActDescChain(_DescChain):
    """The "activation" descriptor chain."""

    def __init__(self, descs, cps_info):
        super().__init__(descs)
        self._out_indices = _OUTPUT_REGISTER_INDICES[cps_info.name.split("_")[0]]

    @property
    def in_np_arrs(self):
        return _find_io_const_np_arrs(self._descs, lambda i: i not in self._out_indices)

    @property
    def out_np_arrs(self):
        return _find_io_const_np_arrs(self._descs, lambda i: i in self._out_indices)


class DescChainArray:
    """The array-like container of descriptor chain."""

    def __init__(self, chains=None):
        self._chains = [] if chains is None else list(chains)

    def __getitem__(self, indices):
        return self._chains[indices]

    def __setitem__(self, indices, value):
        self._chains[indices] = value

    def __len__(self):
        return len(self._chains)

    def append(self, chain):
        assert isinstance(chain, (CtrlDescChain, ParamDescChain, ActDescChain))
        self._chains.append(chain)

    @property
    def ctrl(self):
        return DescChainArray([x for x in self._chains if isinstance(x, CtrlDescChain)])

    @property
    def param(self):
        return DescChainArray([x for x in self._chains if isinstance(x, ParamDescChain)])

    @property
    def act(self):
        return DescChainArray([x for x in self._chains if isinstance(x, ActDescChain)])

    def __add__(self, other):
        assert isinstance(other, DescChainArray)
        return DescChainArray(self._chains + other._chains)

    @property
    def nbytes(self):
        return sum([x.count_of_u32 for x in self._chains]) * 4
