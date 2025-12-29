# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Rename for loop var to simple name."""
from tvm import tir
from tvm.ir.supply import NameSupply


class _Renamer(tir.StmtExprMutator):
    def __init__(self, var_names):
        self.name_seq = [chr(i) for i in range(ord("z"), ord("i") - 1, -1)]
        self.name_supply = NameSupply()
        for name in var_names:
            self.name_supply.reserve_name(name)
        super().__init__()

    def _get_var_name(self):
        var_name = self.name_seq[-1]

        if self.name_supply.contains_name(var_name):
            var_name = self.name_supply.fresh_name(var_name)

        return var_name

    def _rename_loop_var(self, for_op: tir.For):
        loop_var = for_op.loop_var
        # if length of loop var name is not exceed 5, do not rename.
        if len(loop_var.name) <= 20:
            return for_op

        var_name = self._get_var_name()
        new_loop_var = tir.Var(var_name, loop_var.dtype, loop_var.span)
        new_body = tir.stmt_functor.substitute(for_op.body, {loop_var: new_loop_var})

        return tir.For(
            new_loop_var,
            for_op.min,
            for_op.extent,
            for_op.kind,
            new_body,
            for_op.thread_binding,
            for_op.annotations,
            for_op.span,
        )

    def visit_for(self, for_op):
        # Here before super() for correct sequence i, j, k.
        new_for = self._rename_loop_var(for_op)
        if new_for != for_op:
            name = self.name_seq.pop()
        ret = super().visit_for(new_for)
        if new_for != for_op:
            self.name_seq.append(name)
        return ret

    def visit_seq_stmt(self, seq_stmt):
        new_seq = []
        for stmt in seq_stmt.seq:
            if isinstance(stmt, tir.For):
                new_seq.append(self._rename_loop_var(stmt))
            else:
                new_seq.append(stmt)

        if new_seq != list(seq_stmt.seq):
            name = self.name_seq.pop()
        ret = super().visit_seq_stmt(tir.SeqStmt(new_seq, seq_stmt.span))
        if new_seq != list(seq_stmt.seq):
            self.name_seq.append(name)
        return ret


@tir.transform.prim_func_pass(opt_level=0)
class RenameForLoopVar:
    """Rename for loop var to simple name."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        var_names = []
        tir.stmt_functor.post_order_visit(
            func.body, lambda v: var_names.append(v.name) if isinstance(v, tir.Var) else None
        )
        return func.with_body(_Renamer(var_names).visit(func.body), span=func.span)
