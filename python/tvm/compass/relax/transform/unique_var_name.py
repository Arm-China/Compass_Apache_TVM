# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Unique variable name."""
from tvm import ir
from tvm.relax.expr_functor import PyExprMutator, mutator
from .utils import is_cps_composite_func


@mutator
class VarRenamer(PyExprMutator):
    """Unique variable name in caller."""

    def __init__(self, params, mod=None):
        super().__init__(mod)
        self.params = params
        self.var2new_var = {}
        self.unique_name = set()
        self.name2num = {}

    def _check_and_update(self, var):
        # Unneed to unique params now.
        if var in self.params:
            return var
        # Update if visited.
        if var in self.var2new_var:
            return self.var2new_var[var]
        var_name = var.name_hint
        if var_name in self.unique_name:
            new_var_name = var_name + "_" + str(self.name2num[var_name])
            new_var = var.__class__(new_var_name, var.struct_info, var.span)
            self.var2new_var[var] = new_var
            self.name2num[var_name] += 1
            return new_var
        else:
            self.unique_name.add(var_name)
            self.name2num[var_name] = 1
            self.var2new_var[var] = var
            return var

    def visit_var_def_(self, var):
        return self._check_and_update(var)

    def visit_dataflow_var_def_(self, var):
        return self._check_and_update(var)

    def visit_var_(self, var):
        return self._check_and_update(var)

    def visit_dataflow_var_(self, var):
        return self._check_and_update(var)


@ir.transform.module_pass(opt_level=0)
class UniqueVarName:
    """Unique variable name."""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        for gvar, func in ir_mod.functions.items():
            if is_cps_composite_func(func):
                continue
            ir_mod[gvar] = VarRenamer(func.params).visit_expr(func)

        return ir_mod
