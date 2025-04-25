# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=too-many-nested-blocks
"""Sink operator "transpose" to make more optimization space."""
import numpy as np
from tvm import relax, ir
from tvm.relax.analysis import get_var2val
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.dpl import wildcard, is_op, rewrite_call
from .utils import is_call


def _get_inverse_axes(axes):
    axes_dict = {axis: i for i, axis in enumerate(axes)}
    ordered_d = dict(sorted(axes_dict.items()))
    inverse_axes = list(ordered_d.values())
    return inverse_axes


@mutator
class TransposeSinker(PyExprMutator):
    """Sinker of operator "transpose"."""

    def __init__(self, var2val, val2var, mod=None):
        super().__init__(mod)
        self.var2val = var2val
        self.val2var = val2var

    def visit_call_(self, call):
        post = super().visit_call_(call)
        args0 = post.args[0]
        args0 = self.var2val[args0] if args0 in self.var2val else args0
        ops = ["nn.relu", "log", "tanh", "sqrt", "rsqrt"]
        if any(is_call(post, x) for x in ops) and is_call(args0, "permute_dims"):
            new_call = relax.Call(post.op, args0.args, post.attrs, span=post.span)
            out = relax.Call(args0.op, [new_call], args0.attrs)
            self.var2val[self.val2var[post]] = out
            return out

        ops = ["quantize", "dequantize"]
        if any(is_call(post, x) for x in ops) and is_call(args0, "permute_dims"):
            new_post_args = [args0.args[0]] + post.args[1:]
            new_call = relax.Call(post.op, new_post_args, post.attrs, span=post.span)
            out = relax.Call(args0.op, [new_call], args0.attrs)
            self.var2val[self.val2var[post]] = out
            return out

        if is_call(post, "clip") and is_call(args0, "permute_dims"):
            args = [args0.args[0]] + post.args[1:]
            new_call = relax.Call(post.op, args, post.attrs, span=post.span)
            out = relax.Call(args0.op, [new_call], args0.attrs)
            self.var2val[self.val2var[post]] = out
            return out

        if is_call(post, "nn.pad") and is_call(args0, "permute_dims"):
            attrs = post.attrs
            pad_width = [int(x) for x in attrs.pad_width]
            if len(pad_width) != 8:
                return post
            pad_widths = [pad_width[:2], pad_width[2:4], pad_width[4:6], pad_width[6:]]
            inverse_axes = _get_inverse_axes(args0.attrs.axes)
            new_pad_width = []
            for idx in inverse_axes:
                new_pad_width += pad_widths[idx]
            new_pad = relax.op.nn.pad(args0.args[0], new_pad_width, post.args[1], attrs.pad_mode)
            out = relax.Call(args0.op, [new_pad], args0.attrs)
            self.var2val[self.val2var[post]] = out
            return out

        ops = ["add", "multiply", "divide", "maximum", "minimum"]
        if any(is_call(post, x) for x in ops) and is_call(args0, "permute_dims"):
            axes = args0.attrs.axes
            if isinstance(post.args[1], relax.Constant):
                constant = post.args[1].data.numpy()
                constant_ndim = len(constant.shape)
                if constant_ndim != len(axes):
                    broadcast_shape = [1] * (len(axes) - constant_ndim) + list(constant.shape)
                    constant = np.reshape(constant, broadcast_shape)
                inverse_axes = _get_inverse_axes(axes)
                new_constant = np.transpose(constant, inverse_axes)
                new_const = relax.const(new_constant, new_constant.dtype)
                new_arith_op = relax.Call(post.op, [args0.args[0], new_const], post.attrs)
                out = relax.Call(args0.op, [new_arith_op], args0.attrs)
                self.var2val[self.val2var[post]] = out
                return out

            args1 = post.args[1]
            args1 = self.var2val[args1] if args1 in self.var2val else args1
            if is_call(args1, "permute_dims") and ir.structural_equal(axes, args1.attrs.axes):
                new_arith_op = relax.Call(post.op, [args0.args[0], args1.args[0]], post.attrs)
                out = relax.Call(args0.op, [new_arith_op], args0.attrs)
                self.var2val[self.val2var[post]] = out
                return out

        return post


class SimplifyConsecutivePermuteDims:
    """permute_dims1(permute_dims0(inp)) --> inp"""

    def __init__(self):
        self.permute_dims0 = is_op("relax.permute_dims")(wildcard())
        self.permute_dims1 = is_op("relax.permute_dims")(self.permute_dims0)
        self.pattern = self.permute_dims1

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            permute_dims0 = matches[self.permute_dims0]
            permute_dims1 = matches[self.permute_dims1]
            axes0 = permute_dims0.attrs.axes
            axes1 = permute_dims1.attrs.axes
            if all(x == y for x, y in zip(axes0, _get_inverse_axes(axes1))):
                return permute_dims0.args[0]
            return expr

        return self.pattern, rewriter


@ir.transform.module_pass(opt_level=0)
class SinkTranspose:
    """Sink operator "transpose" and simplify consecutive transpose."""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        func = ir_mod["main"]
        var2val = get_var2val(func)
        val2var = {v: k for k, v in var2val.items()}

        update_func = TransposeSinker(var2val, val2var).visit_expr(func)
        update_func = rewrite_call(*SimplifyConsecutivePermuteDims().pr, update_func)
        ir_mod["main"] = update_func
        update_mod = relax.transform.RemoveUnusedOutputs()(ir_mod)

        return update_mod
