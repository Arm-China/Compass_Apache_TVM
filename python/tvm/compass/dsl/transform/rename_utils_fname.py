# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Rename buffer var with function name in allocate const node."""
from tvm import tir, ir


class UpdateUtilsFnameMutator(tir.StmtExprMutator):
    """
    If subroutine-call happens, prim_func call a global_var.
    However the global_var is not the object contained in module.
    They only share the same name_hint.
    This mutator will correct it, and update utils_fname with kernel_fname
    as prefix.
    """

    def __init__(self, global_vars, kernel_fname, utils_fnames):
        super().__init__()
        self._global_vars = {}
        self.kernel_fname = kernel_fname
        self.utils_fnames = utils_fnames
        for gvar in global_vars:
            name = str(gvar.name_hint)
            self._global_vars[name] = gvar

    def visit_call(self, call):
        """visit call to correct op"""
        update_call = super().visit_call(call)
        if isinstance(call.op, ir.GlobalVar):
            name = str(call.op.name_hint)
            if name in self.utils_fnames:
                update_name = self.kernel_fname + "_" + name
                if update_name in self._global_vars:
                    op = self._global_vars[update_name]
                    update_call = tir.Call(
                        update_call.dtype, op, update_call.args, update_call.span
                    )

        return update_call


@ir.transform.module_pass(opt_level=0)
class RenameUtilsFnames:
    """Rename utils_fname  with kernel_fname as prefix."""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        # get kernel_fname, utils_fnames
        def get_fname(mod):
            utils_fnames = []
            kernel_fname = ""
            for func in mod.functions.values():
                if func.attrs["tir.is_entry_func"]:
                    kernel_fname = func.attrs["global_symbol"]
                else:
                    utils_fnames.append(func.attrs["global_symbol"])
            return kernel_fname, utils_fnames

        kernel_fname, utils_fnames = get_fname(ir_mod)

        # update util_fname and gvar with : kernel_fname +"_"+ func_name
        for gvar, func in ir_mod.functions.items():
            if not func.attrs["tir.is_entry_func"]:
                update_name = kernel_fname + "_" + gvar.name_hint
                ir_mod[update_name] = func.with_attr("global_symbol", update_name)
                del ir_mod[gvar]

        # update call with update_utils_fname
        gvars = list(ir_mod.functions.keys())
        mutator = UpdateUtilsFnameMutator(gvars, kernel_fname, utils_fnames)
        for gvar, func in ir_mod.functions.items():
            ir_mod[gvar] = func.with_body(mutator.visit(func.body), span=func.span)
        return ir_mod
