# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
# pylint: disable=invalid-name
"""The Compass extended TIR transform passes."""
from . import _ffi_api


def SimplifyBufferIndex():
    """Simplify indices of "BufferLoad" and "BufferStore" by removing useless iteration variables
    before applying pass "tir.StorageFlatten".

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass.
    """
    return _ffi_api.SimplifyBufferIndex()


def InjectDma(target):
    """Replace the whole attribute statement node whose key is "pragma_compass_dma_copy" to external
    call of Compass C low level DMA interfaces.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectDma(target)
