# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=invalid-name
"""The AIPU Compass extended Relax transform passes."""
import tvm
from tvm import relax


def FuseTuple(target: str = "aipu_compass", entry_name: str = "main") -> tvm.ir.transform.Pass:
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
    return relax.transform._ffi_api.FuseTuple(target, entry_name)
