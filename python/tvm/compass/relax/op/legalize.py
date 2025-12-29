# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
# pylint: disable=invalid-name
"""Legalize of Zhouyi Compass extended operators."""
from tvm import topi, tir, te
from tvm.relax import BlockBuilder, Call, Expr
from tvm.relax.transform.legalize_ops.common import register_legalize


@register_legalize("relax.fake_quant_with_min_max_vars")
def _fake_quant_with_min_max_vars(bb: BlockBuilder, call: Call) -> Expr:
    inp = call.args[0]
    narrow_range = call.attrs.narrow_range
    num_bits = call.attrs.num_bits
    minimum = call.attrs.minimum
    maximum = call.attrs.maximum

    if minimum > 0:
        min_adj = 0
        max_adj = maximum - minimum
    elif maximum < 0:
        min_adj = minimum - maximum
        max_adj = 0
    else:
        scale = (maximum - minimum) / ((1 << num_bits) - 1)
        min_adj = scale * round(minimum / scale)
        max_adj = maximum + min_adj - minimum

    if narrow_range:
        qmin = 1
    else:
        qmin = 0
    qmax = (1 << num_bits) - 1
    scale = (qmax - qmin) / (max_adj - min_adj)

    def _compute(x: te.Tensor, scale: float, qmax: int, min_adj: float, max_adj: float):
        scale = tir.const(scale, "float32")
        qmax = tir.const(qmax, "int32")
        min_adj = tir.const(min_adj, "float32")
        max_adj = tir.const(max_adj, "float32")
        return topi.clip((topi.clip(topi.round(scale * x), 0, qmax) / scale), min_adj, max_adj)

    return bb.call_te(_compute, inp, scale, qmax, min_adj, max_adj)
