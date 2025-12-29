# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Substitute variable which could not be negtive to size var."""
from tvm import tir
from .utils import get_normalized_condition


class _MergeForWhere(tir.StmtExprMutator):
    def __init__(self, block_name):
        super().__init__()
        self.var = []
        self.condition = None
        self.block_name = block_name

    def visit_for(self, for_op):
        kind = for_op.kind
        if kind == tir.ForKind.VECTORIZED:
            return for_op

        self.var.append(for_op.loop_var)
        ret = super().visit_for(for_op)
        if self.condition is not None:
            ret = tir.For(
                ret.loop_var,
                ret.min,
                tir.Max(0, tir.Min(self.condition, ret.extent)),
                ret.kind,
                ret.body,
                ret.thread_binding,
                ret.annotations,
                ret.span,
            )
            self.condition = None
        self.var.pop()
        return ret

    def visit_block_realize(self, op):
        ret = super().visit_block_realize(op)
        if self.block_name and ret.block.name_hint != self.block_name:
            return ret

        # No predicate is T.bool(True), which is IntImm(1).
        if not isinstance(ret.predicate, tir.IntImm):
            var = self.var[-1]
            self.condition = get_normalized_condition(var, op.predicate)
        return tir.BlockRealize(ret.iter_values, True, ret.block, ret.span)


@tir.transform.prim_func_pass(opt_level=0)
class MergeForWhere:
    """Merge where to For node for element-wise schedule."""

    def __init__(self, block_name=None) -> None:
        self.block_name = block_name

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        return func.with_body(_MergeForWhere(self.block_name).visit(func.body), span=func.span)
