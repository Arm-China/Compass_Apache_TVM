# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=unused-argument, keyword-arg-before-vararg
"""The official part of IR APIs."""
import contextlib
from itertools import product
from tvm import ir, tir
from tvm.script import tir as T
from .base import register_ir_api
from .utils import VALID_ATTRS


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


@register_ir_api
def tag(node, attr):
    """Tag attribute message to input node, and optimize it in later pass."""
    msg = f'The arg "attr" expect one of {VALID_ATTRS.keys()}, but got: "{attr}".'
    assert attr in VALID_ATTRS.keys(), msg
    msg = f'The arg "node" expect a expr of S.API, bug got: "{type(node)}".'
    assert isinstance(node, tir.Call) and node.op == ir.Op.get("tir.call_extern"), msg
    name = node.args[0].value
    msg = f'In attr "{attr}", the arg "node" expect one of {VALID_ATTRS[attr]}, but got: "{name}".'
    assert name in VALID_ATTRS[attr], msg
    return tir.call_extern(node.dtype, "tag", node, attr)


@register_ir_api
def _py_tag(node, attr):
    return node


__all__ = (
    "meta_var",
    "vectorized",
    "block",
    "grid",
    "tag",
)
