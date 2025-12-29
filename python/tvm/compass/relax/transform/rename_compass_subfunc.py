# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Rename Compass subfunc."""
from tvm import relax, ir
from tvm.relax.expr_functor import PyExprMutator, mutator
from .utils import is_compass_func


@mutator
class FuncUpdater(PyExprMutator):
    """Update Compass subfunc name in caller."""

    def __init__(self, mod=None):
        super().__init__(mod)
        self._mod = mod
        self.var2new_var = {}
        self._idx = 0

    def visit_call_(self, call):
        ret = super().visit_call_(call)

        if not isinstance(ret.op, relax.GlobalVar):
            return ret
        if not is_compass_func(self._mod[ret.op]):
            return ret

        if ret.op in self.var2new_var:
            new_var = self.var2new_var[ret.op]
            return relax.Call(new_var, ret.args, ret.attrs, ret.sinfo_args, ret.span)

        new_name = "tvm_compass_subfunc" + str(self._idx)
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
class RenameCompassSubfunc:
    """Rename Compass subfunc."""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        for gvar, func in ir_mod.functions.items():
            if is_compass_func(func):
                continue
            new_func, gvar2new_gvar = FuncUpdater(ir_mod).visit(func)
            ir_mod[gvar] = new_func
            for gvar_del, new_gvar in gvar2new_gvar.items():
                func_del = ir_mod[gvar_del]
                del ir_mod[gvar_del]
                ir_mod[new_gvar] = func_del.with_attr("global_symbol", new_gvar.name_hint)

        return ir_mod
