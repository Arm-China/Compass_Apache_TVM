# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=too-many-nested-blocks
"""Sink dequantize in deq(const) for folding inner constant to outer reshape/perm."""
from tvm import relax, ir
from tvm.relax.analysis import get_var2val
from tvm.relax.expr_functor import PyExprMutator, mutator
from .utils import is_call
from ..config import AipuCompassConfig


@mutator
class DequantizeSinker(PyExprMutator):
    """Sinker of operator "dequantize"."""

    def __init__(self, var2val, val2var, mod=None):
        super().__init__(mod)
        self.var2val = var2val
        self.val2var = val2var
        # perm(reshape(deq(const))) --> deq(perm(reshape(const)))
        # first step: reshape(deq(const)) --> deq(reshape(const))
        # reshape(const) is a middle value.
        self.middle_value = []

    def visit_call_(self, call):
        post = super().visit_call_(call)
        dequant = self.var2val.get(post.args[0], post.args[0])
        ops = ["reshape", "permute_dims"]
        if not any(is_call(post, x) for x in ops) and not is_call(dequant, "dequantize"):
            return post

        deq_inp = self.var2val.get(dequant.args[0], dequant.args[0])
        if not isinstance(deq_inp, relax.Constant) and deq_inp not in self.middle_value:
            return post

        attrs = dequant.attrs
        axis = attrs.axis
        inp_shape = self.builder_.normalize(dequant).struct_info.shape
        dim_value = int(inp_shape[axis])
        if is_call(post, "reshape"):
            out_shape = [int(x) for x in self.builder_.normalize(post).struct_info.shape]
            if not all(x in [1, dim_value] for x in out_shape):
                return post
            new_x = relax.Call(post.op, [deq_inp, post.args[1]], post.attrs)
            new_axis = out_shape.index(dim_value)
        else:
            axes = [int(x) for x in post.attrs.axes]
            new_x = relax.Call(post.op, [deq_inp], post.attrs)
            new_axis = axes.index(axis)

        new_dq = relax.op.qdq.dequantize(
            new_x, dequant.args[1], dequant.args[2], new_axis, attrs.out_dtype
        )
        self.var2val[self.val2var[post]] = new_dq
        self.middle_value.append(new_x)
        return new_dq


@ir.transform.module_pass(opt_level=0)
class SinkDequantize:
    """Sink dequantize in deq(const) for folding inner constant to outer reshape/perm."""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        if AipuCompassConfig.get().common["compat_quantized_model"] != "true":
            return ir_mod

        func = ir_mod["main"]
        var2val = get_var2val(func)
        val2var = {v: k for k, v in var2val.items()}

        update_func = DequantizeSinker(var2val, val2var, ir_mod).visit_expr(func)
        ir_mod["main"] = update_func
        update_mod = relax.transform.RemoveUnusedOutputs()(ir_mod)

        return update_mod
