# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Exchange the constant operand of commutative operations to the right side."""
from tvm import ir, tir
from .utils import is_broadcast_const, is_const_pred


_default_funcs = ("vadd", "vmul", "vand", "vor", "vxor")


class _Mutator(tir.StmtExprMutator):
    def _mutate_default(self, call):
        x, y = call.args[2:4]

        if not is_broadcast_const(x) and not is_const_pred(x):
            return call

        return tir.Call(call.dtype, call.op, (call.args[:2] + [y, x] + call.args[4:]), call.span)

    def visit_call(self, call):
        ret = super().visit_call(call)

        if ret.op != ir.Op.get("tir.call_extern"):
            return ret

        func_name = ret.args[0].value
        if func_name in _default_funcs:
            return self._mutate_default(ret)

        return ret


@tir.transform.prim_func_pass(opt_level=0)
class ExchangeConstantToRight:
    """Exchange the constant operand of commutative operations to the right side."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        return func.with_body(_Mutator().visit(func.body), span=func.span)
