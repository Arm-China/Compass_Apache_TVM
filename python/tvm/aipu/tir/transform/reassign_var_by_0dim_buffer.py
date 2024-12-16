# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Support variable reassignment by converting the variable to a 0-dim buffer."""
from collections import defaultdict
from tvm import ir, tir
from tvm.aipu import script as S


class _Analyser(tir.StmtExprVisitor):
    def __init__(self):
        super().__init__()
        self._var2lets = defaultdict(list)

    def visit_let_stmt(self, let_stmt):
        self._var2lets[let_stmt.var].append(let_stmt)
        super().visit_let_stmt(let_stmt)

    def get_rebind_vars(self, func):
        """Traverse the given PrimFunc, and collect the variables which are
        reassigned, only them need to be converted to 0-dim buffer."""
        self.visit(func.body)

        param_vars = []
        local_vars = []
        for var, lets in self._var2lets.items():
            if var in func.params:
                param_vars.append(var)
            elif len(lets) > 1:
                local_vars.append(var)
        return param_vars, local_vars


def _cast_store(value):
    if value.dtype == "bool":
        return S.cast(value, "int8")
    else:
        return value


def _cast_load(value, var):
    if var.dtype == "bool":
        return S.cast(value, "bool")
    else:
        return value


def _create_0dim_buffer(var):
    var_constructor = tir.SizeVar if isinstance(var, tir.SizeVar) else tir.Var
    dtype = "int8" if var.dtype == "bool" else var.dtype
    data = var_constructor(var.name, ir.PointerType(ir.PrimType(dtype), "local"))
    return tir.decl_buffer((1,), dtype, var.name, data=data)


class _ParamRewriter(tir.StmtExprMutator):
    def __init__(self, rebind_vars):
        super().__init__()
        self._rebind_vars = rebind_vars
        self._var2buffer = {}

    def visit_var(self, var):
        if var not in self._var2buffer:
            return var
        return _cast_load(tir.BufferLoad(self._var2buffer[var], (0,), None, var.span), var)

    def visit_let_stmt(self, let_stmt):
        ret = super().visit_let_stmt(let_stmt)

        if ret.var not in self._var2buffer:
            return ret
        buf_store = tir.BufferStore(
            self._var2buffer[ret.var], _cast_store(ret.value), (0,), None, ret.span
        )
        return tir.SeqStmt((buf_store, ret.body))

    def rewrite(self, func_body):
        """Responsible for processing the reassigned function parameters."""
        # 1. Create the corresponding 0-dim buffers.
        self._var2buffer = {x: _create_0dim_buffer(x) for x in self._rebind_vars}

        # 2. Change all let statements to buffer store and all references to buffer load.
        ret = self.visit(func_body)

        # 3. Add buffer store to initialize the 0-dim buffer with the parameter value.
        buf_inits = tuple(
            tir.BufferStore(buf, _cast_store(var), (0,)) for var, buf in self._var2buffer.items()
        )
        ret = tir.SeqStmt(buf_inits + (ret,))

        # 4. Add allocation statements.
        for _, buf in self._var2buffer.items():
            decl_buf = tir.DeclBuffer(buf, ret)
            ret = tir.Allocate(
                buf.data,
                buf.dtype,
                buf.shape,
                tir.const(1, "bool"),
                decl_buf,
                {"reassigned_param_var": True},
            )

        return ret


class _LocalRewriter(tir.StmtExprMutator):
    def __init__(self, rebind_vars):
        super().__init__()
        self._rebind_vars = rebind_vars
        self._var2buffer = {}

    def visit_var(self, var):
        if var not in self._rebind_vars:
            return var
        assert var in self._var2buffer, f'The variable "{var}" used before definition.'
        return _cast_load(tir.BufferLoad(self._var2buffer[var], (0,), None, var.span), var)

    def visit_let_stmt(self, let_stmt):
        var = let_stmt.var
        if var not in self._rebind_vars:
            return super().visit_let_stmt(let_stmt)

        if var in self._var2buffer:
            # Indicate it isn't the 1st let statement of this variable, so just
            # replace it to a buffer store statement.
            ret = super().visit_let_stmt(let_stmt)
            buf_store = tir.BufferStore(
                self._var2buffer[var], _cast_store(ret.value), (0,), None, ret.span
            )
            return tir.SeqStmt((buf_store, ret.body))

        # 1. Create the corresponding 0-dim buffers.
        buf = _create_0dim_buffer(var)
        self._var2buffer[var] = buf

        # 2. Change all other let statements to buffer store and all references to buffer load.
        ret = super().visit_let_stmt(let_stmt)

        # 3. Replace the current let statement to buffer store statement.
        ret = tir.SeqStmt(
            (tir.BufferStore(buf, _cast_store(ret.value), (0,), None, ret.span), ret.body)
        )

        # 4. Add allocation statement.
        return tir.Allocate(
            buf.data,
            buf.dtype,
            buf.shape,
            tir.const(1, "bool"),
            tir.DeclBuffer(buf, ret),
            {"reassigned_local_var": True},
        )


@tir.transform.prim_func_pass(opt_level=0)
class ReassignVarBy0DimBuffer:
    """Support variable reassignment by converting the variable to a 0-dim buffer."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        rebind_param_vars, rebind_local_vars = _Analyser().get_rebind_vars(func)
        new_body = func.body
        if len(rebind_param_vars) != 0:
            new_body = _ParamRewriter(rebind_param_vars).rewrite(new_body)
        if len(rebind_local_vars) != 0:
            new_body = _LocalRewriter(rebind_local_vars).visit(new_body)
        return func.with_body(new_body, span=func.span)
