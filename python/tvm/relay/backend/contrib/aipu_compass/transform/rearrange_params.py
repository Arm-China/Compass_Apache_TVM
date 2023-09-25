# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
# pylint: disable=bad-super-call
"""Rearrange params of some AIPU subgraphs."""
import tvm
from tvm import relay


class Rearranger(relay.ExprMutator):
    """
    Rearrange params of AIPU subgraphs.
    """

    def __init__(self, mod):
        super().__init__()
        self.mod = mod
        self.fix_gfuncs_dict = {}
        self.main_params_dict = {var: i for i, var in enumerate(mod["main"].params)}

    def _check(self, call):
        if len(call.args) == 1 or len(call.args) != len(self.main_params_dict.keys()):
            return False

        for arg in call.args:
            if arg not in self.main_params_dict.keys():
                return False

        return True

    def _rearrange_gfuncs_params(self):
        for gvar in self.mod.get_global_vars():
            if gvar.name_hint in self.fix_gfuncs_dict.keys():
                args_order = self.fix_gfuncs_dict[gvar.name_hint]
                gfunc = self.mod[gvar]
                assert len(args_order) == len(gfunc.params)
                params_dict = dict(zip(args_order, gfunc.params))
                sorted_params_dict = dict(sorted(params_dict.items()))
                new_params = list(sorted_params_dict.values())
                new_func = relay.Function(
                    new_params, gfunc.body, gfunc.ret_type, gfunc.type_params, gfunc.attrs
                )
                self.mod.update_func(gvar, new_func)

        return self.mod

    def visit_call(self, call):
        ret = super().visit_call(call)

        if (
            isinstance(ret, relay.Call)
            and hasattr(ret.op, "name_hint")
            and ret.op.name_hint.startswith("tvmgen_default_aipu_compass")
            and self._check(ret)
        ):
            args_order = [self.main_params_dict[arg] for arg in ret.args]
            if sorted(args_order) == args_order:
                return ret
            else:
                new_args = self.mod["main"].params
                self.fix_gfuncs_dict[ret.op.name_hint] = args_order
                return relay.Call(ret.op, new_args, ret.attrs)

        return ret

    def __call__(self):
        new_main_func = self.visit(self.mod["main"])
        if not self.fix_gfuncs_dict:
            return self.mod

        self.mod.update_func(self.mod.get_global_var("main"), new_main_func)
        update_mod = self._rearrange_gfuncs_params()

        return update_mod


@tvm.ir.transform.module_pass(opt_level=0)
class RearrangeParams:
    """
    Rearrange params of AIPU subgraphs if order mismatch with main_func.
    """

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """try transform module"""
        update_mod = Rearranger(mod)()

        return relay.transform.InferType()(update_mod)
