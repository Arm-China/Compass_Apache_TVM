# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Make the Ramp nodes only appear as the index of buffer load/store."""
from tvm import tir, DataType


class _Canonicalizer(tir.StmtExprMutator):
    def __init__(self):
        super().__init__()
        self._var2ramp = {}

    def visit_var(self, var):
        return self._var2ramp.get(var, var)

    def visit_let_stmt(self, let_stmt):
        if not isinstance(let_stmt.value, tir.Ramp):
            return super().visit_let_stmt(let_stmt)

        var = let_stmt.var
        ramp = let_stmt.value
        new_var = tir.Var(var.name, DataType(var.dtype).element_of, var.span)
        new_value = ramp.base
        # Propagate Ramp information from the vector variable to the relevant buffer load/store.
        self._var2ramp[var] = tir.Ramp(new_var, ramp.stride, ramp.lanes, ramp.span)

        ret = super().visit_let_stmt(let_stmt)
        return tir.LetStmt(new_var, new_value, ret.body, ret.span)


@tir.transform.prim_func_pass(opt_level=0)
class CanonicalizeRamp:
    """Make the Ramp nodes only appear as the index of buffer load/store."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        return func.with_body(_Canonicalizer().visit(func.body), span=func.span)
