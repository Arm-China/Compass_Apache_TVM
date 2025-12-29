# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Rename buffer var with function name in allocate const node."""
from tvm import tir
from tvm.script.ir_builder.base import IRBuilder


class ConstRenamer(tir.StmtExprMutator):
    """Rename buffer var with function name in allocate const node."""

    def __init__(self, func_name):
        super().__init__()
        self.func_name = func_name

    def visit_allocate_const(self, op):
        var = op.buffer_var
        new_name = self.func_name + "_" + var.name
        IRBuilder.name(new_name, var)
        return super().visit_allocate_const(op)


@tir.transform.prim_func_pass(opt_level=0)
class RenameConstBufferVar:
    """Rename buffer var with function name in allocate const node."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        func_name = func.attrs["global_symbol"]
        return func.with_body(ConstRenamer(func_name).visit(func.body), span=func.span)
