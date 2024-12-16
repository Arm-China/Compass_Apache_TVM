# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Uniquify var name in advance."""
from tvm import tir
from tvm.ir.supply import NameSupply
from tvm.script.ir_builder.base import IRBuilder


class _NameUniquifier(tir.StmtExprMutator):
    def __init__(self):
        super().__init__()
        self.name_supply = NameSupply()
        self.visited = set()

    def visit_var(self, var):
        ret = super().visit_var(var)
        if ret in self.visited:
            return ret

        name = ret.name
        if self.name_supply.contains_name(name):
            new_name = self.name_supply.fresh_name(name)
            IRBuilder.name(new_name, ret)
        else:
            self.name_supply.reserve_name(name)
        self.visited.add(ret)
        return ret


@tir.transform.prim_func_pass(opt_level=0)
class UniquifyVarName:
    """Uniquify var name in advance."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        return func.with_body(_NameUniquifier().visit(func.body), span=func.span)
