# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Eliminate common get local id."""
from tvm import tir
from tvm.aipu import script as S


class _Eliminator(tir.StmtExprMutator):
    def __init__(self):
        super().__init__()
        self.var_tid = tir.SizeVar("tid", "int32")
        self.origin_var = None
        self.need_eliminate = False

    def visit_var(self, var):
        if var == self.origin_var:
            return self.var_tid
        return super().visit_var(var)

    def visit_attr_stmt(self, op):
        if isinstance(op.node, tir.IterVar):
            iter_var = op.node
            if (
                iter_var.iter_type == tir.IterVar.ThreadIndex
                and iter_var.thread_tag == "threadIdx.x"
            ):
                self.origin_var = iter_var.var
                self.need_eliminate = True
                return self.visit_stmt(op.body)
        return super().visit_attr_stmt(op)


@tir.transform.prim_func_pass(opt_level=0)
class EliminateGetLocalID:
    """Eliminate common get local id."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        eliminator = _Eliminator()
        new_body = eliminator.visit(func.body)
        if eliminator.need_eliminate:
            new_body = tir.LetStmt(eliminator.var_tid, S.get_local_id(), new_body)
        return func.with_body(new_body, span=func.span)
