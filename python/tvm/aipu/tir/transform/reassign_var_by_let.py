# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Support variable reassignment by let statement and "tir.reassign"."""
from tvm import tir


class _ParamMutator(tir.StmtExprMutator):
    def __init__(self):
        super().__init__()
        self._buf_var2var = {}

    def visit_buffer_load(self, buf_load):
        ret = super().visit_buffer_load(buf_load)

        if buf_load.buffer.data not in self._buf_var2var:
            return ret

        var = self._buf_var2var[buf_load.buffer.data]
        assert var is not None, "Use before define!"
        return var

    def visit_buffer_store(self, buf_store):
        ret = super().visit_buffer_store(buf_store)

        buf_var = buf_store.buffer.data
        if buf_var not in self._buf_var2var:
            return ret

        var = self._buf_var2var[buf_var]
        if var is None:
            # Indicate it's the initialization statement of the 0-dim buffer.
            var = ret.value
            assert isinstance(var, tir.Var) and var.name == buf_var.name
            self._buf_var2var[buf_var] = var
            return tir.Evaluate(0)  # The simplest way to delete a statement.

        # Indicate it's the other reassign statements.
        return tir.Evaluate(tir.reassign(var, ret.value))

    def visit_decl_buffer(self, decl_buffer):
        ret = super().visit_decl_buffer(decl_buffer)

        if ret.buffer.data not in self._buf_var2var:
            return ret
        return ret.body  # Delete the corresponding buffer declaration statement.

    def visit_allocate(self, op):
        if "reassigned_param_var" not in op.annotations:
            return super().visit_allocate(op)

        assert op.buffer_var not in self._buf_var2var, "Redefined!"
        self._buf_var2var[op.buffer_var] = None
        ret = super().visit_allocate(op)
        return ret.body  # Delete the allocation statement.


class _LocalMutator(tir.StmtExprMutator):
    def __init__(self):
        super().__init__()
        self._buf_var2var_init_value = {}

    def visit_buffer_load(self, buf_load):
        ret = super().visit_buffer_load(buf_load)

        if buf_load.buffer.data not in self._buf_var2var_init_value:
            return ret

        var, init_value = self._buf_var2var_init_value[buf_load.buffer.data]
        assert init_value is not None, "Use before define!"
        return var

    def visit_buffer_store(self, buf_store):
        ret = super().visit_buffer_store(buf_store)

        buf_var = buf_store.buffer.data
        if buf_var not in self._buf_var2var_init_value:
            return ret

        var, _ = self._buf_var2var_init_value[buf_var]
        if var is None:
            # Indicate it's the initialization statement of the 0-dim buffer.
            var_constructor = tir.SizeVar if isinstance(buf_var, tir.SizeVar) else tir.Var
            var = var_constructor(buf_var.name, buf_var.type_annotation.element_type.dtype)
            self._buf_var2var_init_value[buf_var] = (var, ret.value)
            return ret  # Replace it to let statement in "visit_seq_stmt".

        # Indicate it's the other reassign statements.
        return tir.Evaluate(tir.reassign(var, ret.value))

    def visit_seq_stmt(self, seq_stmt):
        ret = super().visit_seq_stmt(seq_stmt)

        stack = []
        for stmt in reversed(ret.seq):
            if isinstance(stmt, tir.BufferStore):
                buf_var = stmt.buffer.data
                if buf_var in self._buf_var2var_init_value:
                    var, init_value = self._buf_var2var_init_value[buf_var]
                    assert init_value is not None, "Uninitialized!"
                    body = tir.stmt_seq(*[stack.pop() for i in range(len(stack))])
                    # Replace the buffer store statement with the let statement.
                    stack.append(tir.LetStmt(var, init_value, body, stmt.span))
                    continue

            stack.append(stmt)

        return tir.stmt_seq(*[stack.pop() for i in range(len(stack))])

    def visit_decl_buffer(self, decl_buffer):
        ret = super().visit_decl_buffer(decl_buffer)

        if ret.buffer.data not in self._buf_var2var_init_value:
            return ret
        return ret.body  # Delete the corresponding buffer declaration statement.

    def visit_allocate(self, op):
        if "reassigned_local_var" not in op.annotations:
            return super().visit_allocate(op)

        assert op.buffer_var not in self._buf_var2var_init_value, "Redefined!"
        self._buf_var2var_init_value[op.buffer_var] = (None, None)
        ret = super().visit_allocate(op)
        return ret.body  # Delete the allocation statement.


@tir.transform.prim_func_pass(opt_level=0)
class ReassignVarByLet:
    """Support variable reassignment by let statement and "tir.reassign".

    Before this pass, the variable reassignment is supported by converting the variable to a 0-dim
    buffer, this pass will converting the 0-dim buffer back to let and "tir.reassign" statement, it
    will make our other backend passes easy to implement.
    Notice that after this pass, the TIR isn't a SSA IR, so the official passes maybe can't work if
    you put them after this pass."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        new_body = _ParamMutator().visit(func.body)
        return func.with_body(_LocalMutator().visit(new_body), span=func.span)
