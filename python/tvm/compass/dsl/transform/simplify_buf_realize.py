# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Simplify buffer realize node"""
from tvm import tir


class Simplifier(tir.StmtExprMutator):
    """Simplify buffer realize node"""

    def __init__(self, ext_buf):
        super().__init__()
        self.ext_buf = ext_buf

    def visit_buffer_realize(self, op):
        buf = op.buffer
        return self.visit(op.body) if buf in self.ext_buf else super().visit_buffer_realize(op)


@tir.transform.prim_func_pass(opt_level=0)
class BufRealizeSimplifier:
    """Simplify buffer realize node"""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        ext_buf = func.buffer_map.values()
        return func.with_body(Simplifier(ext_buf).visit(func.body), span=func.span)
