# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=unused-argument, keyword-arg-before-vararg
"""The official part of IR APIs."""
import contextlib
from itertools import product
from tvm.script import tir as T
from .base import register_ir_api


meta_var = register_ir_api(T.meta_var)


@register_ir_api
def _py_meta_var(value, *args, **kwargs):
    return value


vectorized = register_ir_api(T.vectorized)
block = register_ir_api(T.block)
grid = register_ir_api(T.grid)


@register_ir_api
def _py_vectorized(start, stop=None, *args, **kwargs):
    if stop is None:
        stop = start
        start = 0
    return range(start, stop)


@register_ir_api
def _py_block(*args, **kwargs):
    return contextlib.suppress()


@register_ir_api
def _py_grid(*extents, **kwargs):
    for iters in product(*[range(x) for x in extents]):
        yield iters


__all__ = (
    "meta_var",
    "vectorized",
    "block",
    "grid",
)
