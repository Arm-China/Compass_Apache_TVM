# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=too-many-nested-blocks
"""Rename AIPU subfunc."""
from tvm import relax, ir
from tvm.relax.expr_functor import PyExprMutator, mutator


@mutator
class MainFuncUpdater(PyExprMutator):
    """Update AIPU subfunc name in main func caller."""

    def __init__(self, mod=None):
        super().__init__(mod)
        self._mod = mod
        self.var2new_var = {}
        self._idx = 0

    def visit_call_(self, call):
        ret = super().visit_call_(call)

        if not isinstance(ret.op, relax.GlobalVar):
            return ret
        func_attrs = self._mod[ret.op].attrs
        if "Codegen" not in func_attrs or func_attrs.Codegen != "aipu_compass":
            return ret

        if ret.op in self.var2new_var:
            new_var = self.var2new_var[ret.op]
            return relax.Call(new_var, ret.args, ret.attrs, ret.sinfo_args, ret.span)

        new_name = "tvm_aipu_compass_subfunc" + str(self._idx)
        self._idx += 1
        new_var = relax.GlobalVar(new_name)
        relax.expr._update_struct_info(new_var, ret.op.struct_info)
        self.var2new_var[ret.op] = new_var
        new_call = relax.Call(new_var, ret.args, ret.attrs, ret.sinfo_args, ret.span)
        return new_call

    def visit(self, func):
        new_func = self.visit_expr(func)
        return new_func, self.var2new_var


@ir.transform.module_pass(opt_level=0)
class RenameAipuSubfunc:
    """Rename AIPU subfunc."""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        new_main, var2new_var = MainFuncUpdater(ir_mod).visit(ir_mod["main"])
        ir_mod["main"] = new_main
        for gvar, func in ir_mod.functions.items():
            if gvar not in var2new_var.keys():
                continue
            ir_mod.remove(gvar)
            new_var = var2new_var[gvar]
            ir_mod[new_var] = func.with_attr("global_symbol", new_var.name_hint)

        return ir_mod
