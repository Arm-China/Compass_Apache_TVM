# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Adjust function parameters that should be variable of the corresponding buffer."""
from tvm import tir


class _Aligner(tir.StmtExprMutator):
    def __init__(self, var_substitute_map):
        super().__init__()
        self._var_substitute_map = var_substitute_map

    def visit_var(self, var):
        return self._var_substitute_map.get(var, var)


@tir.transform.prim_func_pass(opt_level=0)
class AlignParamVarWithBuffer:
    """Replace function parameters that associated with buffer by the variable
    of the corresponding buffer, so that we can remove the dependency of pass
    "MakeUnpackedAPI"."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        substitute_map = {x: func.buffer_map[x].data for x in func.params if x in func.buffer_map}
        substitute_map = {k: v for k, v in substitute_map.items() if k != v}
        if len(substitute_map) == 0:
            return func

        new_params = [substitute_map.get(x, x) for x in func.params]
        new_buf_map = {substitute_map.get(k, k): v for k, v in func.buffer_map.items()}
        new_body = _Aligner(substitute_map).visit(func.body)
        return tir.PrimFunc(new_params, new_body, func.ret_type, new_buf_map, func.attrs, func.span)
