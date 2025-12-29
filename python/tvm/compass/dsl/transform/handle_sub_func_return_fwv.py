# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Transform sub function to support returning flexible width vector."""
from tvm import tir, ir
from .utils import is_global_var, is_builtin, is_ret
from .. import script as S
from ..utils import is_hw_native_vdtype


def _is_fwv_vdtype(dtype):
    return not (dtype.is_bool or dtype.is_scalar or is_hw_native_vdtype(dtype))


class _Mutator(tir.StmtExprMutator):
    def __init__(self, param_ret):
        super().__init__()
        self.param_ret = param_ret
        self._tmp_var_id = 0
        self._need_insert_lets = []

    def _insert_let(self, stmt, num):
        while num > 0:
            let = self._need_insert_lets.pop()
            new_body = tir.SeqStmt([let.body, stmt])
            stmt = tir.LetStmt(let.var, let.value, new_body, let.span)
            num -= 1
        return stmt

    def visit_let_stmt(self, let_stmt):
        before = len(self._need_insert_lets)
        new_value = self.visit_expr(let_stmt.value)
        num = len(self._need_insert_lets) - before
        new_body = self.visit_stmt(let_stmt.body)

        if new_value == let_stmt.value and new_body == let_stmt.body:
            return let_stmt

        out = tir.LetStmt(let_stmt.var, new_value, new_body, let_stmt.span)
        return self._insert_let(out, num)

    def _mutate_func(self, call):
        dtype = call.dtype
        tmp_var = tir.Var(f"tmp_var{self._tmp_var_id}", dtype)
        self._tmp_var_id += 1
        new_args = list(call.args) + [tmp_var.addr.as_ptr(dtype.element_of)]
        new_call = tir.Call("void", call.op, new_args, call.span)
        body = tir.Evaluate(new_call)
        value = getattr(S, str(dtype))(0)
        self._need_insert_lets.append(tir.LetStmt(tmp_var, value, body))
        return tmp_var

    def visit_call(self, call):
        ret = super().visit_call(call)
        if not _is_fwv_vdtype(ret.dtype):
            return ret

        if is_global_var(ret):
            return self._mutate_func(ret)

        if is_ret(ret):
            return S.vstore(ret.args[0], self.param_ret)
        return ret

    def visit_evaluate(self, evaluate):
        src_value = evaluate.value
        before = len(self._need_insert_lets)
        new_value = self.visit_expr(src_value)
        if new_value == src_value:
            return evaluate

        num = len(self._need_insert_lets) - before
        out = self._insert_let(tir.Evaluate(new_value, evaluate.span), num)
        if is_builtin(new_value, "vstore") and is_ret(src_value):
            ret_none_call = tir.Call("void", ir.Op.get("tir.ret"), [])
            return tir.SeqStmt([out, tir.Evaluate(ret_none_call)])
        return out


@tir.transform.prim_func_pass(opt_level=0)
class HandleSubFuncReturnFWV:
    """Transform sub function to support returning flexible width vector.
    Do transform in caller as follows:
        def func():                            def func():
            va = subfunc0()              -->      tmp_var0 = S.i32x16(0)
            vb = subfunc1() + subfunc2()          subfunc0(..., tmp_var0.addr)
                                                  va = tmp_var0
                                                  tmp_var1 = S.i32x16(0)
                                                  subfunc1(..., tmp_var1.addr)
                                                  tmp_var2 = S.i32x16(0)
                                                  subfunc2(..., tmp_var2.addr)
                                                  vb = tmp_var1 + tmp_var2

    Do transform in callee as follows:
        def sub_func(params) -> S.i32x16:      def sub_func(params, ret:S.ptr) -> None:
            va = xxx                     -->       va = xxx
            return va                              S.vstore(va, ret)
                                                   return
    """

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        new_params = list(func.params)
        param_ret_ptr = None
        ret_type = func.ret_type
        if isinstance(ret_type, ir.PrimType) and _is_fwv_vdtype(ret_type.dtype):
            # Append a "ret" arg in callee function and set ret_type to None.
            elem_dtype = ret_type.dtype.element_of
            param_ret = tir.Var("ret", ir.PointerType(ir.PrimType(elem_dtype), "local"))
            param_ret_ptr = tir.Pointer(elem_dtype, "local", param_ret, name="ret")
            new_params.append(param_ret)
            ret_type = None

        new_body = _Mutator(param_ret_ptr).visit(func.body)
        return tir.PrimFunc(new_params, new_body, ret_type, func.buffer_map, func.attrs, func.span)
