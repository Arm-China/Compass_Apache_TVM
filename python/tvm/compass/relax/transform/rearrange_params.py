# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Rearrange params of some Compass subgraphs."""
from tvm import relax, ir
from tvm.relax.expr_functor import PyExprMutator, mutator
from .utils import is_compass_func


@mutator
class Rearranger(PyExprMutator):
    """
    Rearrange params of Compass subgraphs.
    """

    def __init__(self):
        super().__init__()
        self.fix_gfuncs_dict = {}
        self.main_params_dict = {}

    def _check(self, call):
        if len(call.args) == 1 or len(call.args) != len(self.main_params_dict.keys()):
            return False

        for arg in call.args:
            if arg not in self.main_params_dict.keys():
                return False

        return True

    def visit_call_(self, call):
        ret = super().visit_call_(call)

        if (
            isinstance(ret, relax.Call)
            and hasattr(ret.op, "name_hint")
            and ret.op.name_hint.startswith("tvm_compass_subfunc")
            and self._check(ret)
        ):
            args_order = [self.main_params_dict[arg] for arg in ret.args]
            if sorted(args_order) == args_order:
                return ret
            else:
                new_args = list(self.main_params_dict.keys())
                self.fix_gfuncs_dict[ret.op] = args_order
                return relax.Call(ret.op, new_args, ret.attrs, ret.sinfo_args, ret.span)

        return ret

    def visit(self, func):
        self.main_params_dict = {var: i for i, var in enumerate(func.params)}
        new_func = self.visit_expr(func)
        return new_func, self.fix_gfuncs_dict


@ir.transform.module_pass(opt_level=0)
class RearrangeParams:
    """Rearrange params of Compass subgraphs if order mismatch with main_func."""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        for gvar, func in ir_mod.functions.items():
            if is_compass_func(func):
                continue
            new_func, fix_gfuncs_dict = Rearranger().visit(func)
            if not fix_gfuncs_dict:
                continue
            ir_mod[gvar] = new_func
            for gvar_, args_order in fix_gfuncs_dict.items():
                gfunc = ir_mod[gvar_]
                assert len(args_order) == len(gfunc.params)
                params_dict = dict(zip(args_order, gfunc.params))
                sorted_params_dict = dict(sorted(params_dict.items()))
                new_params = list(sorted_params_dict.values())
                new_func = relax.Function(
                    new_params, gfunc.body, gfunc.ret_struct_info, attrs=gfunc.attrs
                )
                ir_mod[gvar_] = new_func

        return ir_mod
