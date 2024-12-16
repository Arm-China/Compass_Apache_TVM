# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Replace the special parameter "r" mark with the appropriate value."""
from tvm import ir, tir
from tvm.aipu.script.ir.utils import PARAM_R_MARK
from ... import script as S
from .utils import is_builtin


class _InlineReviser(tir.StmtExprMutator):
    def visit_call(self, call):
        ret = super().visit_call(call)

        if ret.op != ir.Op.get("tir.call_extern") or PARAM_R_MARK not in ret.args:
            return ret

        new_args = [S.cast(0, ret.dtype) if PARAM_R_MARK == x else x for x in ret.args]
        return tir.Call(ret.dtype, ret.op, new_args, ret.span)


class _AssignReviser(tir.StmtExprMutator):
    def visit_let_stmt(self, let_stmt):
        ret = super().visit_let_stmt(let_stmt)

        if (
            not isinstance(ret.value, tir.Call)
            or ret.value.op != ir.Op.get("tir.call_extern")
            or PARAM_R_MARK not in ret.value.args
        ):
            return ret

        call = ret.value
        new_args = [tir.precodegen(ret.var) if PARAM_R_MARK == x else x for x in call.args]
        new_call = tir.Call(call.dtype, call.op, new_args, call.span)
        return tir.LetStmt(ret.var, new_call, ret.body, ret.span)

    def visit_call(self, call):
        ret = super().visit_call(call)

        if not is_builtin(ret, "vstore"):
            return ret

        value, ptr, stride, mask = ret.args[1:]
        if (
            not isinstance(value, tir.Call)
            or value.op != ir.Op.get("tir.call_extern")
            or PARAM_R_MARK not in value.args
        ):
            return ret

        new_args = []
        for arg in value.args:
            if PARAM_R_MARK == arg:
                arg = tir.Call(value.dtype, value.op, ("vload", ptr, stride, mask), ret.span)
            new_args.append(arg)

        new_value = tir.Call(value.dtype, value.op, new_args, value.span)
        return tir.Call(ret.dtype, ret.op, ("vstore", new_value, ptr, stride, mask), ret.span)


@tir.transform.prim_func_pass(opt_level=0)
class ReviseParamR:
    """Replace the special parameter "r" mark with the appropriate value."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        new_body = _InlineReviser().visit(_AssignReviser().visit(func.body))
        return func.with_body(new_body, span=func.span)
