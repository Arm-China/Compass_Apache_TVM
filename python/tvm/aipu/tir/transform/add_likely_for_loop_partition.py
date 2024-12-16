# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Wrap the condition of the "if" statement with "tir.likely"."""
from tvm import ir, tir


class _Rewriter(tir.StmtExprMutator):
    def visit_if_then_else(self, ite):
        ret = super().visit_if_then_else(ite)

        condition = ret.condition
        if isinstance(condition, tir.Call) and condition.op == ir.Op.get("tir.likely"):
            return ret
        return tir.IfThenElse(tir.likely(condition), ret.then_case, ret.else_case)


@tir.transform.prim_func_pass(opt_level=0)
class AddLikelyForLoopPartition:
    """Wrap the condition of the "if" statement with "tir.likely".

    Note: This pass should be after tvm.tir.transform.LowerOpaqueBlock,
          This pass should be before tvm.tir.transform.LoopPartition.
    """

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        return func.with_body(_Rewriter().visit(func.body), span=func.span)
