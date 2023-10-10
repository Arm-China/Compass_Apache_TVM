# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
# pylint: disable=bad-super-call
"""Rearrange names of the AIPU subgraphs."""
import tvm
from tvm import relay


class Rearranger(relay.ExprMutator):
    """
    Rename main_idx of AIPU Compass Function.
    """

    def __init__(self, mod):
        super().__init__()
        self.mod = mod
        self.gfunc_names = []
        self.func_gvar_dict = {}

    def _rename_gfuncs_name(self, main_func):
        gfuncs_dict = {}
        for gvar in self.mod.get_global_vars():
            if gvar.name_hint in self.func_gvar_dict:
                gfunc = self.mod[gvar]
                new_func = relay.Function(
                    gfunc.params, gfunc.body, gfunc.ret_type, gfunc.type_params
                )
                new_gvar = self.func_gvar_dict[gvar.name_hint]
                for k, v in gfunc.attrs.items():
                    if k == "global_symbol":
                        v = new_gvar.name_hint
                    new_func = new_func.with_attr(k, v)
                gfuncs_dict.update({new_gvar: new_func})
            elif gvar.name_hint != "main":
                gfuncs_dict.update({gvar: self.mod[gvar]})

        update_mod = tvm.IRModule.from_expr(main_func, gfuncs_dict)
        return update_mod

    def visit_call(self, call):
        ret = super().visit_call(call)

        if (
            isinstance(ret, relay.Call)
            and hasattr(ret.op, "name_hint")
            and ret.op.name_hint.startswith("tvmgen_default_aipu_compass")
        ):
            func_name = ret.op.name_hint
            if func_name not in self.gfunc_names:
                self.gfunc_names.append(func_name)
            name_pre, main_idx = func_name.rsplit("_", 1)
            expect_main_idx = len(self.gfunc_names) - 1
            if int(main_idx) == expect_main_idx:
                return ret
            new_func_name = "_".join([name_pre, str(expect_main_idx)])
            new_gvar = relay.GlobalVar(new_func_name, ret.op.checked_type)
            self.func_gvar_dict.update({func_name: new_gvar})
            return relay.Call(new_gvar, ret.args, ret.attrs, ret.type_args, ret.span)

        return ret

    def __call__(self):
        new_main_func = self.visit(self.mod["main"])
        if not self.func_gvar_dict:
            return self.mod

        self.mod.update_func(self.mod.get_global_var("main"), new_main_func)
        update_mod = self._rename_gfuncs_name(new_main_func)
        return update_mod


@tvm.ir.transform.module_pass(opt_level=0)
class RearrangeNames:
    """
    Rearrange names of AIPU subgraphs if id in the name is not ordered.
    """

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """try transform module"""
        update_mod = Rearranger(mod)()

        return relay.transform.InferType()(update_mod)
