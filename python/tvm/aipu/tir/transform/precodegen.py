# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Generate the code of some specific TIR expressions in advance."""
from tvm import ir, tir
from .utils import is_all_true_pred


class _Gener(tir.StmtExprMutator):
    def _gen_precodegen(self, call):
        arg = call.args[0]
        if isinstance(arg, tir.StringImm):
            return call

        if isinstance(arg, tir.Var):
            return tir.precodegen(arg.name, dtype=call.dtype)

        raise RuntimeError(f'Unknown precodegen: "{call}".')

    def _gen_vmov_pred(self, call):
        pred_value = call.args[1]
        if not isinstance(pred_value, tir.IntImm):
            return call

        func_name = call.args[0].value
        ret_str = f"{func_name}(0x{pred_value.value:08X})"
        if is_all_true_pred(call):
            ret_str = f"ALL_TRUE_{func_name[-1]}"
        return tir.precodegen(ret_str, dtype=call.dtype)

    def visit_call(self, call):
        ret = super().visit_call(call)

        if ret.op == ir.Op.get("tir.precodegen"):
            return self._gen_precodegen(ret)

        if ret.op != ir.Op.get("tir.call_extern"):
            return ret

        func_name = ret.args[0].value
        if func_name.startswith("__vmov_"):
            return self._gen_vmov_pred(ret)

        return ret


@tir.transform.prim_func_pass(opt_level=0)
class Precodegen:
    """Generate the code of some specific TIR expressions in advance."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        return func.with_body(_Gener().visit(func.body), span=func.span)
