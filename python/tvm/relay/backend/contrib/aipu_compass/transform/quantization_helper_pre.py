# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
""" Rewrite module by pattern """
# pylint: disable=bad-super-call
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    is_constant,
    is_op,
    wildcard,
)
from .hint_pattern_rewrite import RewriteDecorator


@RewriteDecorator
class AdaptDenseAdd(DFPatternCallback):
    """dense + add -> dense + bias_add"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        dense = is_op("nn.dense")(wildcard(), wildcard())
        add = is_op("add")(dense, wildcard())
        quantize = is_op("qnn.quantize")(add, is_constant(), is_constant())
        self.pattern = quantize

    def callback(self, pre, post, node_map):
        quantize = node_map[self.pattern][0]
        add = quantize.args[0]
        dense, weight = add.args
        dense_shape = dense.checked_type.shape
        weight_shape = weight.checked_type.shape
        if len(weight_shape) > 1:
            return post
        if weight_shape[-1] != dense_shape[-1]:
            return post
        bias_add = relay.nn.bias_add(dense, weight, axis=-1)
        return relay.Call(
            quantize.op,
            [bias_add] + quantize.args[1:],
            quantize.attrs,
            quantize.type_args,
            quantize.span,
        )


@tvm.ir.transform.module_pass(opt_level=0)
class QuantizationHelperPre:
    """
    Function to rewrite module by pattern
    """

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """try transform module"""

        update_mod = relay.transform.InferType()(mod)
        infer_type = relay.transform.InferType()
        simplify_expr = relay.transform.SimplifyExpr()

        patterns = [
            AdaptDenseAdd(before_passes=[simplify_expr, infer_type]),
        ]
        for pattern in patterns:
            update_mod = pattern(update_mod)  # pylint: disable=not-callable

        return update_mod
