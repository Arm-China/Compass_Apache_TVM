# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Legalize var name so that it could be parsed by text parser."""
from tvm import relax
from tvm.relax.expr_functor import PyExprMutator, mutator


def verify_name(name):
    name = name.replace("-", "_")
    name = name.replace(".", "_")
    name = name.replace(":", "_")
    return name


@mutator
class VarNameRewriter(PyExprMutator):
    """Legalize var name."""

    def __init__(self, params, mod=None):
        super().__init__(mod)
        self.params = params
        self.var2new_var = {}

    def visit_var_def_(self, var):
        if var in self.params:
            verified_name = verify_name(var.name_hint)
            if var.name_hint != verified_name:
                new_var = relax.Var(verified_name, var.struct_info)
                self.var2new_var[var] = new_var
                return new_var
        return var

    def visit_dataflow_var_def_(self, var):
        if var in self.params:
            verified_name = verify_name(var.name_hint)
            if var.name_hint != verified_name:
                new_var = relax.Var(verified_name, var.struct_info)
                self.var2new_var[var] = new_var
                return new_var
        return var

    def visit_var_(self, var):
        return self.var2new_var.get(var, var)

    def visit_dataflow_var_(self, var):
        return self.var2new_var.get(var, var)


@relax.transform.function_pass(opt_level=0)
class LegalizeVarName:
    """Legalize var name to rewrite illegal character in var name."""

    def transform_function(self, func, mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        return VarNameRewriter(func.params).visit_expr(func)
