# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Substitute variable which could not be negtive to size var."""
from tvm import tir
from .utils import is_builtin


class _Mutator(tir.StmtExprMutator):
    def __init__(self):
        super().__init__()
        self._var_substitute_map = {}

    def visit_var(self, var):
        return self._var_substitute_map.get(var, var)

    def visit_let_stmt(self, let_stmt):
        new_value = self.visit_expr(let_stmt.value)

        new_var = var = let_stmt.var
        if is_builtin(new_value, ("get_local_size", "get_local_id")):
            new_var = tir.SizeVar(var.name, var.dtype)
            self._var_substitute_map[var] = new_var

        new_body = self.visit_stmt(let_stmt.body)
        if new_value == let_stmt.value and new_body == let_stmt.body:
            return let_stmt
        return tir.LetStmt(new_var, new_value, new_body, let_stmt.span)


@tir.transform.prim_func_pass(opt_level=0)
class SubstituteSizeVar:
    """Substitute variable which could not be negtive to size var."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        return func.with_body(_Mutator().visit(func.body), span=func.span)
