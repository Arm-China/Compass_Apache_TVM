# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=invalid-name
"""Detect and add DMA pragma."""
import functools
from tvm import tir
from tvm.arith.pattern import detect_linear_equation


class _Rewriter(tir.StmtExprMutator):
    def _collect_indice(self, stmt):
        if stmt.buffer.data.type_annotation.storage_scope in ["", "global"]:
            stmt = stmt.value
        return stmt.indices[0]

    def visit_for(self, stmt):
        origin = stmt
        loop_vars = []
        extent = []

        # Step1: get buffer store and buffer load
        while isinstance(stmt, tir.For):
            if stmt.kind == tir.ForKind.VECTORIZED:
                return origin
            loop_vars.append(stmt.loop_var)
            extent.append(stmt.extent)
            stmt = stmt.body
        extent.append(1)

        # If stmt is not bufferstore, it is possible there is for in its body.
        # Proceed to recursive stmt.
        if not isinstance(stmt, tir.BufferStore):
            return super().visit_for(origin)
        store = stmt
        stmt = stmt.value
        if not isinstance(stmt, tir.BufferLoad):
            return origin
        load = stmt

        # Step2: Analysis scope and indice, only add pragma when scope is not equal
        #        and loop_vars equal indice_vars.
        if store.buffer.data.type_annotation == load.buffer.data.type_annotation:
            return origin

        indice = self._collect_indice(store)
        indice_coeff = detect_linear_equation(indice, loop_vars)
        # If successful match
        if len(indice_coeff) != 0:
            for i in range(len(loop_vars)):
                # Here only deal with DMA1D
                if indice_coeff[i] != functools.reduce(lambda a, b: a * b, extent[i + 1 :]):
                    return origin
            return tir.AttrStmt(origin.loop_var, "pragma_compass_dma_copy", 1, origin)

        return origin


@tir.transform.prim_func_pass(opt_level=0)
class AddDMAPragma:
    """Detect and add DMA pragma."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        return func.with_body(_Rewriter().visit(func.body), span=func.span)
