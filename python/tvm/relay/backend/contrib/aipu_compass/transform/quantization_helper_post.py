# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
""" Rewrite module by pattern """
# pylint: disable=bad-super-call, not-callable
import numpy as np
import tvm
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass.transform.evaluate_zero_free_args_call import (
    vmobj_to_list,
)
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    is_constant,
    is_op,
    wildcard,
)
from .hint_pattern_rewrite import RewriteDecorator


def min_value(dtype):
    return tvm.tir.op.min_value(dtype).value


def max_value(dtype):
    return tvm.tir.op.max_value(dtype).value


def const_eval(exp):
    mod = tvm.IRModule.from_expr(exp)
    mod = relay.transform.FoldConstant()(mod)

    eval_args_func = relay.create_executor(
        kind="vm", mod=mod, device=tvm.cpu(0), target="llvm"
    ).evaluate()
    eval_val = vmobj_to_list(eval_args_func())
    return eval_val


@RewriteDecorator
class EliminateMaximum(DFPatternCallback):
    """maximum(x, x.dtype.min_value) -> x"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        maximum = is_op("maximum")(wildcard(), is_constant())
        self.pattern = maximum

    def callback(self, pre, post, node_map):
        maximum = node_map[self.pattern][0]
        data, const = maximum.args
        dtype = maximum.checked_type.dtype

        max_const = np.amax(const.data.numpy())
        if max_const <= min_value(dtype):
            return data
        else:
            return post


@RewriteDecorator
class AdaptMaximum(DFPatternCallback):
    """conv + bias_add + maximum + requantize -> conv + bias_add + requantize + clip"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        q_conv = is_op("qnn.conv2d")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        add = is_op("nn.bias_add")(q_conv, is_constant())
        maximum = is_op("maximum")(add, is_constant())
        requantize = is_op("qnn.requantize")(
            maximum, is_constant(), is_constant(), is_constant(), is_constant()
        )
        self.pattern = requantize

    def callback(self, pre, post, node_map):
        requantize = node_map[self.pattern][0]
        maximum = requantize.args[0]
        data, const = maximum.args
        requantize_dtype = requantize.attrs.out_dtype
        requantize_args = requantize.args[1:]

        if requantize_dtype not in ("uint8", "int8"):
            return post
        new_data = relay.Call(
            requantize.op,
            [data] + requantize_args,
            requantize.attrs,
            requantize.type_args,
            requantize.span,
        )
        new_const = relay.Call(
            requantize.op,
            [const] + requantize_args,
            requantize.attrs,
            requantize.type_args,
            requantize.span,
        )
        new_const = const_eval(new_const).numpy()
        max_const = np.amax(new_const)
        min_const = np.amin(new_const)
        if max_const != min_const:
            return post
        return relay.clip(new_data, max_const, max_value(requantize_dtype))


@RewriteDecorator
class AdaptClipCast(DFPatternCallback):
    """requantize + clip + cast -> requantize"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        req = is_op("qnn.requantize")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant()
        ).has_attr({"out_dtype": "int32"})
        clip = is_op("clip")(req)
        cast = is_op("cast")(clip)
        self.pattern = cast

    def callback(self, pre, post, node_map):
        cast = node_map[self.pattern][0]
        clip = cast.args[0]
        requantize = clip.args[0]

        cast_dtype = cast.checked_type.dtype
        clip_attrs = clip.attrs
        req_attrs = requantize.attrs

        if clip_attrs.a_min == min_value(cast_dtype) and clip_attrs.a_max == max_value(cast_dtype):
            return relay.qnn.op.requantize(
                *requantize.args,
                axis=req_attrs.axis,
                rounding=req_attrs.rounding,
                compute_dtype=req_attrs.compute_dtype,
                out_dtype=cast_dtype,
            )
        else:
            return post


@tvm.ir.transform.module_pass(opt_level=0)
class QuantizationHelperPost:
    """
    Function to rewrite module by pattern
    """

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """try transform module"""

        update_mod = relay.transform.InferType()(mod)
        fold_constant = relay.transform.FoldConstant()

        patterns = [
            AdaptMaximum(),
            EliminateMaximum(before_passes=fold_constant),
            AdaptClipCast(),
        ]
        for pattern in patterns:
            update_mod = pattern(update_mod)

        return update_mod
