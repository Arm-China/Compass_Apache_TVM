# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=unused-argument
"""The axis part of IR APIs."""
from tvm.script import tir as T
from .base import register_ir_api

spatial = register_ir_api(T.axis.spatial)
reduce = register_ir_api(T.axis.reduce)
remap = register_ir_api(T.axis.remap)

S = spatial  # pylint: disable=invalid-name
R = reduce  # pylint: disable=invalid-name


@register_ir_api
def _py_spatial(dom, binding, *args, **kwargs):
    return binding


@register_ir_api
def _py_reduce(dom, binding, *args, **kwargs):
    return binding


@register_ir_api
def _py_remap(kinds, bindings, *args, **kwargs):
    return bindings[0] if len(bindings) == 1 else bindings


__all__ = ("spatial", "reduce", "remap")
