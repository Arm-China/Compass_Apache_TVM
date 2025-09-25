# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=invalid-name
"""The Zhouyi Compass extended Relax transform passes."""
import tvm
from . import _ffi_api


def FuseTuple(target: str = "compass", entry_name: str = "main") -> tvm.ir.transform.Pass:
    """Fuse Tuple and TupleGetItem to target

    Parameters
    ----------
    target: str
        The byoc target name
    entry_name: str
        The entry name

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.FuseTuple(target, entry_name)
