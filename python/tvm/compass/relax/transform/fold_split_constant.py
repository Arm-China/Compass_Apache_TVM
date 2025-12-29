# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Fold split constant."""
import numpy as np
from tvm import relax, tir, ir
from tvm.relax.analysis import get_var2val
from tvm.relax.expr_functor import PyExprVisitor, visitor
from tvm.relax.expr_functor import PyExprMutator, mutator
from .utils import is_call


@visitor
class Visitor(PyExprVisitor):
    """Visit and get fold split map."""

    def __init__(self):
        self.fold_split = {}

    def visit_call_(self, call):
        if is_call(call, "split") and isinstance(call.args[0], relax.Constant):
            data = call.args[0].data.numpy()
            ind_or_sec = call.attrs.indices_or_sections
            if isinstance(ind_or_sec, tir.IntImm):
                ind_or_sec = int(ind_or_sec)
            else:
                ind_or_sec = [int(x) for x in ind_or_sec]
            splits = np.split(data, ind_or_sec, axis=call.attrs.axis)
            self.fold_split[call] = [relax.const(x, x.dtype) for x in splits]
        return super().visit_call_(call)

    def get_fold_split(self, expr):
        """Get call to split constant data map."""
        self.visit_expr(expr)
        return self.fold_split


@mutator
class Mutator(PyExprMutator):
    """Mutate tuple get item from fold split."""

    def __init__(self, var2val, fold_split, mod=None):
        super().__init__(mod)
        self.var2val = var2val
        self.fold_split = fold_split

    def visit_tuple_getitem_(self, tgi):
        tup_value = self.var2val.get(tgi.tuple_value)
        if tup_value and tup_value in self.fold_split:
            return self.fold_split[tup_value][tgi.index]
        return super().visit_tuple_getitem_(tgi)


@ir.transform.module_pass(opt_level=0)
class FoldSplitConstant:
    """Fold split constant."""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        for gvar, func in ir_mod.functions.items():
            fold_split = Visitor().get_fold_split(func)
            while len(fold_split) > 0:
                ir_mod[gvar] = Mutator(get_var2val(func), fold_split).visit_expr(func)
                ir_mod = relax.transform.FoldConstant()(ir_mod)
                func = ir_mod[gvar]
                fold_split = Visitor().get_fold_split(func)
        return relax.transform.RemoveUnusedOutputs()(ir_mod)
