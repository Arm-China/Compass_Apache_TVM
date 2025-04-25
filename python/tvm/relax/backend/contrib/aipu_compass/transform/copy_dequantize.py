# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=too-many-nested-blocks
"""Copy dequantize."""
from tvm import relax, ir
from tvm.relax.analysis import get_var2val
from tvm.relax.expr_functor import PyExprVisitor, visitor
from tvm.relax.expr_functor import PyExprMutator, mutator
from .utils import is_call
from ..config import AipuCompassConfig


@visitor
class Visitor(PyExprVisitor):
    """Analyze and get reuse dequantize."""

    def __init__(self, var2val):
        super().__init__()
        self.var2val = var2val
        self.dequant_usage = {}

    def visit_call_(self, call):
        dequant = self.var2val.get(call.args[0], call.args[0])
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
        dequant = self.var2val.get(ret.args[0], ret.args[0])
        if is_call(dequant, "dequantize") and dequant in self.reuse_dequant:
            copy_dequant = relax.Call(dequant.op, dequant.args, dequant.attrs)
            return relax.Call(ret.op, [copy_dequant] + ret.args[1:], ret.attrs)
        return ret


@ir.transform.module_pass(opt_level=0)
class CopyDequantize:
    """Make a copy for reuse dequant."""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        if AipuCompassConfig.get().common["compat_quantized_model"] != "true":
            return ir_mod

        func = ir_mod["main"]
        var2val = get_var2val(func)
        reuse_dequant = Visitor(var2val).get_reuse_dequant(func)
        update_func = Mutator(var2val, reuse_dequant, ir_mod).visit_expr(func)
        ir_mod["main"] = update_func
        update_mod = relax.transform.RemoveUnusedOutputs()(ir_mod)

        return update_mod
