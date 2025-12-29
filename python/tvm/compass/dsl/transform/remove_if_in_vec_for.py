# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Remove the if statement in for statement."""
from tvm import tir
from .utils import get_normalized_condition


class _Remover(tir.StmtExprMutator):
    def visit_for(self, for_op):
        ret = super().visit_for(for_op)

        if ret.kind == tir.ForKind.VECTORIZED:
            if_stmt = ret.body
            var = ret.loop_var
            lanes = ret.extent

            # Before:
            # for i in S.vectorized(n % 8):
            #     xxx
            # After:
            # T.attr(i, "origin_vectorize_extent", 8)
            # for i in S.vectorized(n % 8):
            #     xxx
            if isinstance(lanes, tir.FloorMod):
                return tir.AttrStmt(var, "origin_vectorize_extent", lanes.b, ret)

            # Before:
            # for i in S.vectorized(8):
            #     if i + k < size
            #         xxx
            # After:
            # T.attr(i, "origin_vectorize_extent", 8)
            # for i in S.vectorized(size - k):
            #     xxx
            if isinstance(if_stmt, tir.IfThenElse):
                extent = get_normalized_condition(var, if_stmt.condition)
                if extent is not None:
                    new_for = tir.For(
                        var,
                        ret.min,
                        tir.Max(0, tir.Min(extent, lanes)),
                        ret.kind,
                        if_stmt.then_case,
                        ret.thread_binding,
                        ret.annotations,
                        ret.span,
                    )
                    return tir.AttrStmt(new_for.loop_var, "origin_vectorize_extent", lanes, new_for)
        else:
            # Before:
            # for i in S.for(8):
            #     if i + k < size
            #         xxx
            # After:
            # for i in S.for(min(8, size - k)):
            #     xxx
            if_stmt = ret.body
            var = ret.loop_var

            if isinstance(if_stmt, tir.IfThenElse):
                extent = get_normalized_condition(var, if_stmt.condition)
                if extent is not None:
                    new_for = tir.For(
                        var,
                        ret.min,
                        tir.Max(0, tir.Min(extent, ret.extent)),
                        ret.kind,
                        if_stmt.then_case,
                        ret.thread_binding,
                        ret.annotations,
                        ret.span,
                    )
                    return new_for
        return ret


@tir.transform.prim_func_pass(opt_level=0)
class RemoveIfInVecFor:
    """Remove the if statement in for statement."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        return func.with_body(_Remover().visit(func.body), span=func.span)
