# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Fold constant value."""
from tvm import ir, tir, DataType
from .utils import is_builtin, is_broadcast_const, is_const_pred


class _Analyzer(tir.StmtExprVisitor):
    def __init__(self):
        super().__init__()
        self._vars = set()

    def visit_call(self, call):
        super().visit_call(call)

        if call.op == ir.Op.get("tir.pointer") and isinstance(call.args[2], tir.Var):
            # Indicate this pointer is created by getting address of a variable.
            self._vars.add(call.args[2])

        if call.op == ir.Op.get("tir.reassign"):
            self._vars.add(call.args[0])

        if is_builtin(call, "inline_asm"):
            self._vars |= set(x for x in call.args[1:] if isinstance(x, tir.Var))

    def get_cannot_fold_vars(self, func):
        self.visit(func.body)
        return self._vars


class _Mutator(tir.StmtExprMutator):
    def __init__(self, cannot_fold_vars):
        super().__init__()
        self._cannot_fold_vars = cannot_fold_vars
        self._var2const = {}

    def visit_var(self, var):
        if var not in self._var2const:
            return var

        const_value = self._var2const[var]
        if is_const_pred(const_value):
            # Make a copy for each user is important for the mask node, it will make the passe
            # "AlignVectorWidthBySplit" and "AlignVectorWidthByPad" work better.
            return tir.Call(const_value.dtype, const_value.op, const_value.args, const_value.span)

        return const_value

    def visit_let_stmt(self, let_stmt):
        new_value = self.visit_expr(let_stmt.value)

        if let_stmt.var not in self._cannot_fold_vars:
            if (
                isinstance(new_value, (tir.IntImm, tir.FloatImm, tir.StringImm))
                or is_broadcast_const(new_value)
                or is_const_pred(new_value)
            ):
                self._var2const[let_stmt.var] = new_value
                return self.visit_stmt(let_stmt.body)

        new_body = self.visit_stmt(let_stmt.body)
        if new_value == let_stmt.value and new_body == let_stmt.body:
            return let_stmt
        return tir.LetStmt(let_stmt.var, new_value, new_body, let_stmt.span)

    def _mutate_vbcast(self, call):
        dtype, x = DataType(call.dtype), call.args[2]

        if dtype.is_bool and isinstance(x, tir.IntImm):
            # The mask is all true when broadcasting a boolean, it is guaranteed by parser.
            return tir.const_pred((bool(x),) * dtype.lanes, span=call.span)

        return call

    def visit_call(self, call):
        ret = super().visit_call(call)

        if ret.op != ir.Op.get("tir.call_extern"):
            return ret

        func_name = ret.args[0].value
        if func_name == "__vbcast":
            return self._mutate_vbcast(ret)

        return ret


@tir.transform.prim_func_pass(opt_level=0)
class FoldConstant:
    """Fold constant value."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        cannot_fold_vars = _Analyzer().get_cannot_fold_vars(func)
        return func.with_body(_Mutator(cannot_fold_vars).visit(func.body), span=func.span)
