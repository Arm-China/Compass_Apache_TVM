# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Copy dequantize."""
from tvm import relax
from tvm.relax.analysis import get_var2val
from tvm.relax.expr_functor import PyExprVisitor, visitor
from tvm.relax.expr_functor import PyExprMutator, mutator
from .utils import is_call
from ..config import CompassConfig


@visitor
class Visitor(PyExprVisitor):
    """Analyze and get reuse dequantize."""

    def __init__(self, var2val):
        super().__init__()
        self.var2val = var2val
        self.dequant_usage = {}

    def visit_call_(self, call):
        call_args = call.args[0].fields if is_call(call, "concat") else call.args
        for arg in call_args:
            dequant = self.var2val.get(arg, arg)
            if is_call(dequant, "dequantize"):
                if dequant in self.dequant_usage:
                    self.dequant_usage[dequant] += 1
                else:
                    self.dequant_usage[dequant] = 1
        return super().visit_call_(call)

    def get_reuse_dequant(self, func):
        """Return reuse dequant list."""
        self.visit_expr(func)
        return [k for k, v in self.dequant_usage.items() if v > 1]


@mutator
class Mutator(PyExprMutator):
    """Make a copy for reuse dequant."""

    def __init__(self, var2val, reuse_dequant, mod=None):
        super().__init__(mod)
        self.var2val = var2val
        self.reuse_dequant = reuse_dequant

    def visit_call_(self, call):
        ret = super().visit_call_(call)
        args = list(ret.args[0].fields) if is_call(call, "concat") else list(ret.args)
        new_args = []
        for arg in args:
            dequant = self.var2val.get(arg, arg)
            if is_call(dequant, "dequantize") and dequant in self.reuse_dequant:
                copy_dequant = relax.Call(dequant.op, dequant.args, dequant.attrs)
                new_args.append(copy_dequant)
            else:
                new_args.append(arg)
        if args == new_args:
            return ret
        new_args = [relax.Tuple(new_args)] if is_call(call, "concat") else new_args
        return relax.Call(ret.op, new_args, ret.attrs)


@relax.transform.function_pass(opt_level=0)
class CopyDequantize:
    """Make a copy for reuse dequant."""

    def transform_function(self, func, mod, pass_ctx):  # pylint: disable=unused-argument
        """Transform the given function and return the result."""
        if CompassConfig.get().common["compat_quantized_model"] != "true":
            return func

        var2val = get_var2val(func)
        reuse_dequant = Visitor(var2val).get_reuse_dequant(func)
        updated_func = Mutator(var2val, reuse_dequant).visit_expr(func)
        updated_func = relax.analysis.remove_all_unused(updated_func)

        return updated_func
