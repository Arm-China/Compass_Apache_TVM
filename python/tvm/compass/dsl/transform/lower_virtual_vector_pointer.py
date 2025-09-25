# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Lower each virtual vector pointer to the corresponding hardware native scalar pointer."""
from tvm import tir, ir
from ..utils import is_hw_native_dtype


def _try_mutate_type(x):
    if isinstance(x, ir.PointerType) and isinstance(x.element_type, ir.PrimType):
        dtype = x.element_type.dtype
        if not (dtype.is_void or is_hw_native_dtype(dtype)):
            return ir.PointerType(ir.PrimType(dtype.element_of), x.storage_scope)

    return x


def _try_mutate_pointer_var(var, var_substitute_map):
    new_type_ann = _try_mutate_type(var.type_annotation)
    if new_type_ann != var.type_annotation:
        new_var = tir.Var(var.name, new_type_ann, var.span)
        var_substitute_map[var] = new_var
        return new_var

    return var


class _Mutator(tir.StmtExprMutator):
    def __init__(self, var_substitute_map):
        super().__init__()
        self._var_substitute_map = var_substitute_map

    def visit_var(self, var):
        return self._var_substitute_map.get(var, var)

    def _mutate_pointer(self, call):
        type_annotation, scope, base, offset = call.args
        dtype = type_annotation.dtype

        if dtype.is_void or is_hw_native_dtype(dtype):
            return call

        # Change the virtual vector pointer to the corresponding hardware native scalar pointer.
        return tir.pointer(dtype.element_of, scope, base, offset * dtype.lanes)

    def visit_let_stmt(self, let_stmt):
        new_value = self.visit_expr(let_stmt.value)
        new_var = _try_mutate_pointer_var(let_stmt.var, self._var_substitute_map)
        new_body = self.visit_stmt(let_stmt.body)

        if new_value == let_stmt.value and new_body == let_stmt.body:
            return let_stmt
        return tir.LetStmt(new_var, new_value, new_body, let_stmt.span)

    def visit_call(self, call):
        ret = super().visit_call(call)

        if ret.op != ir.Op.get("tir.pointer"):
            return ret

        return self._mutate_pointer(ret)


@tir.transform.prim_func_pass(opt_level=0)
class LowerVirtualVectorPointer:
    """Lower each virtual vector pointer to the corresponding hardware native scalar pointer."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        var_substitute_map = {}
        new_params = tuple(_try_mutate_pointer_var(x, var_substitute_map) for x in func.params)
        new_body = _Mutator(var_substitute_map).visit(func.body)
        new_ret_type = _try_mutate_type(func.ret_type)

        return tir.PrimFunc(
            new_params, new_body, new_ret_type, func.buffer_map, func.attrs, func.span
        )
