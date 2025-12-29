# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Revert some Compass TIR nodes back to the corresponding standard representation."""
from tvm import tir, ir
from .utils import is_all_true_pred


class _Mutator(tir.StmtExprMutator):
    def __init__(self):
        super().__init__()
        self._var2buffer = {}
        self._forced_revert_load_store_vars = []

    def _get_or_create_buffer(self, var, buf_dtype):
        if var not in self._var2buffer:
            self._var2buffer[var] = tir.decl_buffer((-1,), buf_dtype, f"{var.name}_buf", data=var)
        return self._var2buffer[var]

    def visit_allocate(self, op):
        if "need_revert_load_store" in op.annotations:
            self._forced_revert_load_store_vars.append(op.buffer_var)
        return super().visit_allocate(op)

    def _mutate_vload(self, call):
        ptr, stride, mask = call.args[1:4]
        ptr_base = ptr.args[2]

        if ptr_base in self._forced_revert_load_store_vars or (
            stride == 1
            and is_all_true_pred(mask)
            and isinstance(ptr_base, tir.Var)
            and isinstance(ptr_base.type_annotation, ir.PointerType)
        ):
            ptr_dtype = ptr.args[0].dtype
            buf = self._get_or_create_buffer(ptr_base, ptr_dtype)
            ptr_offset = ptr.args[3]
            return tir.BufferLoad(buf, (ptr_offset,), span=call.span)

        return call

    def _mutate_vstore(self, call):
        ptr, stride, mask = call.args[2:5]
        ptr_base = ptr.args[2]

        if ptr_base in self._forced_revert_load_store_vars or (
            stride == 1
            and is_all_true_pred(mask)
            and isinstance(ptr_base, tir.Var)
            and isinstance(ptr_base.type_annotation, ir.PointerType)
        ):
            ptr_dtype = ptr.args[0].dtype
            buf = self._get_or_create_buffer(ptr_base, ptr_dtype)
            value = call.args[1]
            ptr_offset = ptr.args[3]
            return tir.BufferStore(buf, value, (ptr_offset,), span=call.span)

        return call

    def visit_call(self, call):
        ret = super().visit_call(call)

        if ret.op != ir.Op.get("tir.call_extern"):
            return ret

        func_name = ret.args[0].value
        if func_name == "vload":
            return self._mutate_vload(ret)
        if func_name == "vstore":
            return self._mutate_vstore(ret)

        return ret

    def visit_evaluate(self, evaluate):
        new_value = self.visit_expr(evaluate.value)
        if isinstance(new_value, tir.Stmt):
            return new_value
        if new_value == evaluate.value:
            return evaluate
        return tir.Evaluate(new_value, evaluate.span)


@tir.transform.prim_func_pass(opt_level=0)
class Revert2Standard:
    """Revert some Compass TIR nodes back to the corresponding standard representation.

    The goal of this pass is simplifying codegen's implementation, only the needed specific nodes
    will be revert back, e.g., "vload/vstore" that without mask and stride will be reverted back to
    "BufferLoad/BufferStore", so it can be codegened to "a[x]" instead of "__vload(a + x)".
    """

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        return func.with_body(_Mutator().visit(func.body), span=func.span)
