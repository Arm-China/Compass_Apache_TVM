# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Statement functor utilities for IR transformations"""
#
# This file has been modified by Arm China team.
#
from .. import ir
from . import _ffi_api, expr as _expr, stmt as _stmt
from .function import PrimFunc
from .expr_functor import ExprVisitor, ExprMutator


def ir_transform(stmt, preorder, postorder, only_enable=None):
    """Recursively visit and transform ir nodes in post DFS order.

    Parameters
    ----------
    stmt : tvm.tir.Stmt
        The input to be transformed.

    preorder: function
        The function called in before recursive mutation
        If preorder returns None, then the transform will proceed to recursive call.
        If preorder returns a not None tvm.tir.Stmt/Expr, the transformer will simply return it and
        won't do further recursion.

    postorder : function
        The function called after recursive mutation.

    only_enable : Optional[List[str]]
        List of types that we only enable.

    Returns
    -------
    result : tvm.tir.Stmt
        The result.
    """
    return _ffi_api.IRTransform(stmt, preorder, postorder, only_enable)  # type: ignore


def post_order_visit(stmt, fvisit):
    """Recursively visit the ir in post DFS order node, apply fvisit
       Each node is guaranteed to be visited only once.

    Parameters
    ----------
    fvisit: function
        The visitor function.
    """
    return _ffi_api.PostOrderVisit(stmt, fvisit)  # type: ignore


def pre_order_visit(stmt, fvisit):
    """Recursive pre-order visit on stmt AST, applying fvisit on each node.
       If fvisit returns False, it won't visit the children of the node.

    Parameters
    ----------
    fvisit: function of the signature Object -> bool
        The visitor function.
    """
    return _ffi_api.PreOrderVisit(stmt, fvisit)  # type: ignore


def substitute(node, vmap):
    """Substitute the var specified by vmap.

    Parameters
    ----------
    node: ObjectRef
        The input.

    vmap : Dict[Var, PrimExpr]
        The variable mapping.

    Returns
    -------
    result : tvm.tir.Stmt
        The result.
    """
    return _ffi_api.Substitute(node, vmap)  # type: ignore


def renew_defs(func: PrimFunc):
    """Re-generate the definition nodes for a TIR, including VarDef, BufferDef.
    This pass works as a simple DeepCopy to duplicate a function with different Vars and
    Buffers but the same behavior

    Parameters
    ----------
    func: PrimFunc
        The input function

    Returns
    -------
    result : PrimFunc
        The new generated func.
    """
    return _ffi_api.RenewDefs(func)  # type: ignore


class StmtFunctor:
    """An abstract visitor over TIR statement.

    Defines the default dispatch over statement and implements memorization.
    """

    def __init__(self):
        self._memo_map = {}

    def visit_stmt(self, stmt):
        """Apply the visitor to a TIR statement."""
        for key in self._memo_map:
            if key is stmt:
                return self._memo_map[stmt]

        if isinstance(stmt, _stmt.LetStmt):
            ret = self.visit_let_stmt(stmt)
        elif isinstance(stmt, _stmt.AttrStmt):
            ret = self.visit_attr_stmt(stmt)
        elif isinstance(stmt, _stmt.IfThenElse):
            ret = self.visit_if_then_else(stmt)
        elif isinstance(stmt, _stmt.For):
            ret = self.visit_for(stmt)
        elif isinstance(stmt, _stmt.While):
            ret = self.visit_while(stmt)
        elif isinstance(stmt, _stmt.Allocate):
            ret = self.visit_allocate(stmt)
        elif isinstance(stmt, _stmt.AllocateConst):
            ret = self.visit_allocate_const(stmt)
        elif isinstance(stmt, _stmt.DeclBuffer):
            ret = self.visit_decl_buffer(stmt)
        elif isinstance(stmt, _stmt.BufferStore):
            ret = self.visit_buffer_store(stmt)
        elif isinstance(stmt, _stmt.BufferRealize):
            ret = self.visit_buffer_realize(stmt)
        elif isinstance(stmt, _stmt.AssertStmt):
            ret = self.visit_assert_stmt(stmt)
        elif isinstance(stmt, _stmt.ProducerStore):
            ret = self.visit_producer_store(stmt)
        elif isinstance(stmt, _stmt.ProducerRealize):
            ret = self.visit_producer_realize(stmt)
        elif isinstance(stmt, _stmt.Prefetch):
            ret = self.visit_prefetch(stmt)
        elif isinstance(stmt, _stmt.SeqStmt):
            ret = self.visit_seq_stmt(stmt)
        elif isinstance(stmt, _stmt.Evaluate):
            ret = self.visit_evaluate(stmt)
        elif isinstance(stmt, _stmt.Block):
            ret = self.visit_block(stmt)
        elif isinstance(stmt, _stmt.BlockRealize):
            ret = self.visit_block_realize(stmt)
        else:
            raise RuntimeError(f"Unhandled case: {type(stmt)}")

        self._memo_map[stmt] = ret

        return ret

    def visit_let_stmt(self, _):
        raise NotImplementedError()

    def visit_attr_stmt(self, _):
        raise NotImplementedError()

    def visit_if_then_else(self, _):
        raise NotImplementedError()

    def visit_for(self, _):
        raise NotImplementedError()

    def visit_while(self, _):
        raise NotImplementedError()

    def visit_allocate(self, _):
        raise NotImplementedError()

    def visit_allocate_const(self, _):
        raise NotImplementedError()

    def visit_decl_buffer(self, _):
        raise NotImplementedError()

    def visit_buffer_store(self, _):
        raise NotImplementedError()

    def visit_buffer_realize(self, _):
        raise NotImplementedError()

    def visit_assert_stmt(self, _):
        raise NotImplementedError()

    def visit_producer_store(self, _):
        raise NotImplementedError()

    def visit_producer_realize(self, _):
        raise NotImplementedError()

    def visit_prefetch(self, _):
        raise NotImplementedError()

    def visit_seq_stmt(self, _):
        raise NotImplementedError()

    def visit_evaluate(self, _):
        raise NotImplementedError()

    def visit_block(self, _):
        raise NotImplementedError()

    def visit_block_realize(self, _):
        raise NotImplementedError()


class StmtVisitor(StmtFunctor):
    """A visitor over TIR statement.

    The default behavior recursively traverses the TIR statement.
    """

    def visit_expr(self, _):
        pass

    def visit_let_stmt(self, let_stmt):
        self.visit_expr(let_stmt.value)
        self.visit_stmt(let_stmt.body)

    def visit_attr_stmt(self, attr_stmt):
        self.visit_expr(attr_stmt.value)
        self.visit_stmt(attr_stmt.body)

    def visit_if_then_else(self, ite):
        self.visit_expr(ite.condition)
        self.visit_stmt(ite.then_case)
        if ite.else_case:
            self.visit_stmt(ite.else_case)

    def visit_for(self, for_op):
        self.visit_expr(for_op.min)
        self.visit_expr(for_op.extent)
        self.visit_stmt(for_op.body)

    def visit_while(self, while_op):
        self.visit_expr(while_op.condition)
        self.visit_stmt(while_op.body)

    def visit_allocate(self, allocate):
        for extent in allocate.extents:
            self.visit_expr(extent)
        self.visit_stmt(allocate.body)
        self.visit_expr(allocate.condition)

    def visit_allocate_const(self, allocate_const):
        for extent in allocate_const.extents:
            self.visit_expr(extent)
        self.visit_stmt(allocate_const.body)

    def visit_decl_buffer(self, decl_buffer):
        self.visit_stmt(decl_buffer.body)

    def visit_buffer_store(self, buf_store):
        self.visit_expr(buf_store.value)
        for idx in buf_store.indices:
            self.visit_expr(idx)
        self.visit_expr(buf_store.predicate)

    def visit_buffer_realize(self, buffer_realize):
        for bound in buffer_realize.bounds:
            self.visit_expr(bound.min)
            self.visit_expr(bound.extent)
        self.visit_expr(buffer_realize.condition)
        self.visit_stmt(buffer_realize.body)

    def visit_assert_stmt(self, assert_stmt):
        self.visit_expr(assert_stmt.condition)
        self.visit_expr(assert_stmt.message)
        self.visit_stmt(assert_stmt.body)

    def visit_producer_store(self, store):
        for idx in store.indices:
            self.visit_expr(idx)
        self.visit_expr(store.value)

    def visit_producer_realize(self, producer_realize):
        for bound in producer_realize.bounds:
            self.visit_expr(bound.min)
            self.visit_expr(bound.extent)
        self.visit_stmt(producer_realize.body)
        self.visit_expr(producer_realize.condition)

    def visit_prefetch(self, prefetch):
        for bound in prefetch.bounds:
            self.visit_expr(bound.min)
            self.visit_expr(bound.extent)

    def visit_seq_stmt(self, seq_stmt):
        for stmt in seq_stmt.seq:
            self.visit_stmt(stmt)

    def visit_evaluate(self, evaluate):
        self.visit_expr(evaluate.value)

    def _visit_buffer_region(self, buffer_region):
        for r in buffer_region.region:
            self.visit_expr(r.min)
            self.visit_expr(r.extent)

    def visit_block(self, block):
        for iv in block.iter_vars:
            self.visit_expr(iv.dom.min)
            self.visit_expr(iv.dom.extent)

        for buf_region in block.reads:
            self._visit_buffer_region(buf_region)

        for buf_region in block.writes:
            self._visit_buffer_region(buf_region)

        for match_buf_region in block.match_buffers:
            self._visit_buffer_region(match_buf_region.source)

        if block.init:
            self.visit_stmt(block.init)
        self.visit_stmt(block.body)

    def visit_block_realize(self, block_realize):
        for expr in block_realize.iter_values:
            self.visit_expr(expr)
        self.visit_expr(block_realize.predicate)
        self.visit_stmt(block_realize.block)


class StmtMutator(StmtFunctor):
    """A mutator over TIR statement.

    The default behavior recursively traverses and reconstructs the TIR statement.
    """

    def visit_expr(self, expr):
        return expr

    def visit_let_stmt(self, let_stmt):
        new_value = self.visit_expr(let_stmt.value)
        new_body = self.visit_stmt(let_stmt.body)
        if new_value == let_stmt.value and new_body == let_stmt.body:
            return let_stmt
        return _stmt.LetStmt(let_stmt.var, new_value, new_body, let_stmt.span)

    def visit_attr_stmt(self, op):
        new_value = self.visit_expr(op.value)
        new_body = self.visit_stmt(op.body)
        if new_value == op.value and new_body == op.body:
            return op
        return _stmt.AttrStmt(op.node, op.attr_key, new_value, new_body, op.span)

    def visit_if_then_else(self, ite):
        new_cond = self.visit_expr(ite.condition)
        new_then = self.visit_stmt(ite.then_case)
        new_else = ite.else_case
        if ite.else_case:
            new_else = self.visit_stmt(ite.else_case)
        if new_cond == ite.condition and new_then == ite.then_case and new_else == ite.else_case:
            return ite
        return _stmt.IfThenElse(new_cond, new_then, new_else, ite.span)

    def visit_for(self, for_op):
        new_min = self.visit_expr(for_op.min)
        new_extent = self.visit_expr(for_op.extent)
        new_body = self.visit_stmt(for_op.body)
        if new_min == for_op.min and new_extent == for_op.extent and new_body == for_op.body:
            return for_op
        return _stmt.For(
            for_op.loop_var,
            new_min,
            new_extent,
            for_op.kind,
            new_body,
            for_op.thread_binding,
            for_op.annotations,
            for_op.span,
        )

    def visit_while(self, while_op):
        new_condition = self.visit_expr(while_op.condition)
        new_body = self.visit_stmt(while_op.body)
        if new_condition == while_op.condition and new_body == while_op.body:
            return while_op
        return _stmt.While(new_condition, new_body, while_op.span)

    def visit_allocate(self, op):
        new_extents = [self.visit_expr(extent) for extent in op.extents]
        new_body = self.visit_stmt(op.body)
        new_cond = self.visit_expr(op.condition)
        if new_extents == list(op.extents) and new_body == op.body and new_cond == op.condition:
            return op
        return _stmt.Allocate(
            op.buffer_var, op.dtype, new_extents, new_cond, new_body, op.annotations, op.span
        )

    def visit_allocate_const(self, op):
        new_extents = [self.visit_expr(extent) for extent in op.extents]
        new_body = self.visit_stmt(op.body)
        if new_extents == list(op.extents) and new_body == op.body:
            return op
        data_or_idx = op.data or op.irmod_storage_idx
        return _stmt.AllocateConst(
            op.buffer_var, op.dtype, new_extents, data_or_idx, new_body, op.annotations, op.span
        )

    def visit_decl_buffer(self, decl_buffer):
        new_body = self.visit_stmt(decl_buffer.body)
        if new_body == decl_buffer.body:
            return decl_buffer
        return _stmt.DeclBuffer(decl_buffer.buffer, new_body, decl_buffer.span)

    def visit_buffer_store(self, buf_store):
        new_value = self.visit_expr(buf_store.value)
        new_indices = [self.visit_expr(idx) for idx in buf_store.indices]
        new_pred = self.visit_expr(buf_store.predicate)
        if (
            new_value == buf_store.value
            and new_indices == list(buf_store.indices)
            and new_pred == buf_store.predicate
        ):
            return buf_store
        return _stmt.BufferStore(buf_store.buffer, new_value, new_indices, new_pred, buf_store.span)

    def _visit_range(self, r):
        new_min = self.visit_expr(r.min)
        new_extent = self.visit_expr(r.extent)
        if new_min == r.min and new_extent == r.extent:
            return r
        return ir.Range.from_min_extent(new_min, new_extent, r.span)

    def visit_buffer_realize(self, op):
        new_bounds = [self._visit_range(bound) for bound in op.bounds]
        new_condition = self.visit_expr(op.condition)
        new_body = self.visit_stmt(op.body)
        if new_bounds == list(op.bounds) and new_condition == op.condition and new_body == op.body:
            return op
        return _stmt.BufferRealize(op.buffer, new_bounds, new_condition, new_body, op.span)

    def visit_assert_stmt(self, op):
        new_condition = self.visit_expr(op.condition)
        new_message = self.visit_expr(op.message)
        new_body = self.visit_stmt(op.body)
        if new_condition == op.condition and new_message == op.message and new_body == op.body:
            return op
        return _stmt.AssertStmt(new_condition, new_message, new_body, op.span)

    def visit_producer_store(self, store):
        new_indices = [self.visit_expr(idx) for idx in store.indices]
        new_value = self.visit_expr(store.value)
        if new_indices == list(store.indices) and new_value == store.value:
            return store
        return _stmt.ProducerStore(store.producer, new_value, new_indices, store.span)

    def visit_producer_realize(self, op):
        new_bounds = [self._visit_range(bound) for bound in op.bounds]
        new_body = self.visit_stmt(op.body)
        new_condition = self.visit_expr(op.condition)
        if new_bounds == list(op.bounds) and new_body == op.body and new_condition == op.condition:
            return op
        return _stmt.ProducerRealize(
            op.producer, new_bounds, new_condition, new_body, op.storage_scope, op.span
        )

    def visit_prefetch(self, prefetch):
        new_bounds = [self._visit_range(bound) for bound in prefetch.bounds]
        if new_bounds == list(prefetch.bounds):
            return prefetch
        return _stmt.Prefetch(prefetch.buffer, new_bounds, prefetch.span)

    def visit_seq_stmt(self, seq_stmt):
        new_seq = [self.visit_stmt(stmt) for stmt in seq_stmt.seq]
        if new_seq == list(seq_stmt.seq):
            return seq_stmt
        return _stmt.SeqStmt(new_seq, seq_stmt.span)

    def visit_evaluate(self, evaluate):
        new_value = self.visit_expr(evaluate.value)
        if new_value == evaluate.value:
            return evaluate
        return _stmt.Evaluate(new_value, evaluate.span)

    def _visit_iter_var(self, iv):
        new_dom = self._visit_range(iv.dom)
        if new_dom == iv.dom:
            return iv
        return _expr.IterVar(new_dom, iv.var, iv.iter_type, iv.thread_tag, iv.span)

    def _visit_buffer_region(self, buffer_region):
        new_region = [self._visit_range(r) for r in buffer_region.region]
        if new_region == list(buffer_region.region):
            return buffer_region
        return _stmt.BufferRegion(buffer_region.buffer, new_region)

    def _visit_match_buffer_region(self, match_buffer_region):
        new_source = self._visit_buffer_region(match_buffer_region.source)
        if new_source == match_buffer_region.source:
            return match_buffer_region
        return _stmt.MatchBufferRegion(match_buffer_region.buffer, new_source)

    def visit_block(self, block):
        new_iter_vars = [self._visit_iter_var(iv) for iv in block.iter_vars]
        new_reads = [self._visit_buffer_region(buf_region) for buf_region in block.reads]
        new_writes = [self._visit_buffer_region(buf_region) for buf_region in block.writes]
        new_match_buffers = [self._visit_match_buffer_region(x) for x in block.match_buffers]

        new_init = self.visit_stmt(block.init) if block.init else block.init
        new_body = self.visit_stmt(block.body)
        if (
            new_iter_vars == list(block.iter_vars)
            and new_reads == list(block.reads)
            and new_writes == list(block.writes)
            and new_match_buffers == list(block.match_buffers)
            and new_init == block.init
            and new_body == block.body
        ):
            return block
        return _stmt.Block(
            new_iter_vars,
            new_reads,
            new_writes,
            block.name_hint,
            new_body,
            new_init,
            block.alloc_buffers,
            new_match_buffers,
            block.annotations,
            block.span,
        )

    def visit_block_realize(self, op):
        new_ivs = [self.visit_expr(iv) for iv in op.iter_values]
        new_pred = self.visit_expr(op.predicate)
        new_block = self.visit_stmt(op.block)
        if new_ivs == list(op.iter_values) and new_pred == op.predicate and new_block == op.block:
            return op
        return _stmt.BlockRealize(new_ivs, new_pred, new_block, op.span)


class StmtExprVisitor(StmtVisitor, ExprVisitor):
    """A visitor over TIR statement and expression.

    The default behavior recursively traverses the TIR statement and expression.
    """

    def visit_expr(self, expr):
        ExprVisitor.visit_expr(self, expr)

    def visit(self, stmt_or_expr):
        if isinstance(stmt_or_expr, _stmt.Stmt):
            self.visit_stmt(stmt_or_expr)
            return
        self.visit_expr(stmt_or_expr)


class StmtExprMutator(StmtMutator, ExprMutator):
    """A mutator over TIR statement and expression.

    The default behavior recursively traverses and reconstructs the TIR statement and expression.
    """

    def visit_expr(self, expr):
        return ExprMutator.visit_expr(self, expr)

    def visit(self, stmt_or_expr):
        if isinstance(stmt_or_expr, _stmt.Stmt):
            return self.visit_stmt(stmt_or_expr)
        return self.visit_expr(stmt_or_expr)
