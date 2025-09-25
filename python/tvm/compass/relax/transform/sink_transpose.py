# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Sink operator "transpose" to make more optimization space."""
import numpy as np
from tvm import relax, ir
from tvm.relax.analysis import get_var2val
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.dpl import rewrite_call
from .utils import is_call, get_inverse_axes
from .pattern_rewrites import SimplifyConsecutivePermuteDims


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
            inverse_axes = get_inverse_axes(args0.attrs.axes)
            new_pad_width = []
            for idx in inverse_axes:
                new_pad_width += pad_widths[idx]
            new_pad = relax.op.nn.pad(args0.args[0], new_pad_width, attrs.pad_mode, attrs.pad_value)
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
                inverse_axes = get_inverse_axes(axes)
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

        if any(is_call(post, x) for x in ["mean", "sum"]) and is_call(args0, "permute_dims"):
            reduce_attrs = post.attrs
            reduce_axes = reduce_attrs.axis
            new_axes = []
            for x in reduce_axes:
                new_axes.append(args0.attrs.axes[x.value])

            reduce_func = getattr(relax.op, post.op.name.split(".")[-1])
            new_reduce = reduce_func(args0.args[0], new_axes, reduce_attrs.keepdims)

            inp_shape = [int(x) for x in post.args[0].struct_info.shape]
            new_reduce_shape = [x for i, x in enumerate(inp_shape) if i not in reduce_axes]
            keepdims = reduce_attrs.keepdims
            if keepdims == 0 and [int(x) for x in post.struct_info.shape] == new_reduce_shape:
                # The axes to be transposed have already been eliminated by the reduce op.
                return new_reduce
            return relax.Call(args0.op, [new_reduce], args0.attrs)

        return post


@relax.transform.function_pass(opt_level=0)
class SinkTranspose:
    """Sink operator "transpose" and simplify consecutive transpose."""

    def transform_function(self, func, mod, pass_ctx):  # pylint: disable=unused-argument
        """Transform the given function and return the result."""
        var2val = get_var2val(func)
        val2var = {v: k for k, v in var2val.items()}
        updated_func = TransposeSinker(var2val, val2var).visit_expr(func)
        updated_func = rewrite_call(*SimplifyConsecutivePermuteDims().pr, updated_func)
        updated_func = relax.analysis.remove_all_unused(updated_func)

        return updated_func
