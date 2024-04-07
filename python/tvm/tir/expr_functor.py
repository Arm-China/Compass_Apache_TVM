# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=invalid-name
"""Expression functor utilities for TIR transformations."""
from .. import ir
from . import expr as _expr


class ExprFunctor:
    """An abstract visitor over TIR expression.

    Defines the default dispatch over expression and implements memorization.
    """

    def __init__(self):
        self._memo_map = {}

    def visit_expr(self, expr):
        """Apply the visitor to a TIR expression."""
        for key in self._memo_map:
            if key is expr:
                return self._memo_map[expr]

        if isinstance(expr, _expr.Var):
            ret = self.visit_var(expr)
        elif isinstance(expr, _expr.SizeVar):
            ret = self.visit_size_var(expr)
        elif isinstance(expr, _expr.BufferLoad):
            ret = self.visit_buffer_load(expr)
        elif isinstance(expr, _expr.ProducerLoad):
            ret = self.visit_producer_load(expr)
        elif isinstance(expr, _expr.Let):
            ret = self.visit_let(expr)
        elif isinstance(expr, _expr.Call):
            ret = self.visit_call(expr)
        elif isinstance(expr, _expr.Add):
            ret = self.visit_add(expr)
        elif isinstance(expr, _expr.Sub):
            ret = self.visit_sub(expr)
        elif isinstance(expr, _expr.Mul):
            ret = self.visit_mul(expr)
        elif isinstance(expr, _expr.Div):
            ret = self.visit_div(expr)
        elif isinstance(expr, _expr.Mod):
            ret = self.visit_mod(expr)
        elif isinstance(expr, _expr.FloorDiv):
            ret = self.visit_floor_div(expr)
        elif isinstance(expr, _expr.FloorMod):
            ret = self.visit_floor_mod(expr)
        elif isinstance(expr, _expr.Min):
            ret = self.visit_min(expr)
        elif isinstance(expr, _expr.Max):
            ret = self.visit_max(expr)
        elif isinstance(expr, _expr.EQ):
            ret = self.visit_eq(expr)
        elif isinstance(expr, _expr.NE):
            ret = self.visit_ne(expr)
        elif isinstance(expr, _expr.LT):
            ret = self.visit_lt(expr)
        elif isinstance(expr, _expr.LE):
            ret = self.visit_le(expr)
        elif isinstance(expr, _expr.GT):
            ret = self.visit_gt(expr)
        elif isinstance(expr, _expr.GE):
            ret = self.visit_ge(expr)
        elif isinstance(expr, _expr.And):
            ret = self.visit_and(expr)
        elif isinstance(expr, _expr.Or):
            ret = self.visit_or(expr)
        elif isinstance(expr, _expr.Reduce):
            ret = self.visit_reduce(expr)
        elif isinstance(expr, _expr.Cast):
            ret = self.visit_cast(expr)
        elif isinstance(expr, _expr.Not):
            ret = self.visit_not(expr)
        elif isinstance(expr, _expr.Select):
            ret = self.visit_select(expr)
        elif isinstance(expr, _expr.Ramp):
            ret = self.visit_ramp(expr)
        elif isinstance(expr, _expr.Broadcast):
            ret = self.visit_broadcast(expr)
        elif isinstance(expr, _expr.Shuffle):
            ret = self.visit_shuffle(expr)
        elif isinstance(expr, _expr.IntImm):
            ret = self.visit_int_imm(expr)
        elif isinstance(expr, _expr.FloatImm):
            ret = self.visit_float_imm(expr)
        elif isinstance(expr, _expr.StringImm):
            ret = self.visit_string_imm(expr)
        elif isinstance(expr, _expr.Any):
            ret = self.visit_any(expr)
        else:
            raise RuntimeError(f"Unhandled case: {type(expr)}")

        self._memo_map[expr] = ret

        return ret

    def visit_var(self, _):
        raise NotImplementedError()

    def visit_size_var(self, _):
        raise NotImplementedError()

    def visit_buffer_load(self, _):
        raise NotImplementedError()

    def visit_producer_load(self, _):
        raise NotImplementedError()

    def visit_let(self, _):
        raise NotImplementedError()

    def visit_call(self, _):
        raise NotImplementedError()

    def visit_add(self, _):
        raise NotImplementedError()

    def visit_sub(self, _):
        raise NotImplementedError()

    def visit_mul(self, _):
        raise NotImplementedError()

    def visit_div(self, _):
        raise NotImplementedError()

    def visit_mod(self, _):
        raise NotImplementedError()

    def visit_floor_div(self, _):
        raise NotImplementedError()

    def visit_floor_mod(self, _):
        raise NotImplementedError()

    def visit_min(self, _):
        raise NotImplementedError()

    def visit_max(self, _):
        raise NotImplementedError()

    def visit_eq(self, _):
        raise NotImplementedError()

    def visit_ne(self, _):
        raise NotImplementedError()

    def visit_lt(self, _):
        raise NotImplementedError()

    def visit_le(self, _):
        raise NotImplementedError()

    def visit_gt(self, _):
        raise NotImplementedError()

    def visit_ge(self, _):
        raise NotImplementedError()

    def visit_and(self, _):
        raise NotImplementedError()

    def visit_or(self, _):
        raise NotImplementedError()

    def visit_reduce(self, _):
        raise NotImplementedError()

    def visit_cast(self, _):
        raise NotImplementedError()

    def visit_not(self, _):
        raise NotImplementedError()

    def visit_select(self, _):
        raise NotImplementedError()

    def visit_ramp(self, _):
        raise NotImplementedError()

    def visit_broadcast(self, _):
        raise NotImplementedError()

    def visit_shuffle(self, _):
        raise NotImplementedError()

    def visit_int_imm(self, _):
        raise NotImplementedError()

    def visit_float_imm(self, _):
        raise NotImplementedError()

    def visit_string_imm(self, _):
        raise NotImplementedError()

    def visit_any(self, _):
        raise NotImplementedError()


class ExprVisitor(ExprFunctor):
    """A visitor over TIR expression.

    The default behavior recursively traverses the TIR expression.
    """

    def visit_var(self, _):
        pass

    def visit_size_var(self, size_var):
        self.visit_var(size_var)

    def visit_buffer_load(self, buf_load):
        for idx in buf_load.indices:
            self.visit_expr(idx)
        self.visit_expr(buf_load.predicate)

    def visit_producer_load(self, producer_load):
        for idx in producer_load.indices:
            self.visit_expr(idx)

    def visit_let(self, let):
        self.visit_expr(let.value)
        self.visit_expr(let.body)

    def visit_call(self, call):
        for arg in call.args:
            self.visit_expr(arg)

    def _visit_binary_op(self, bin_op):
        self.visit_expr(bin_op.a)
        self.visit_expr(bin_op.b)

    def visit_add(self, op):
        self._visit_binary_op(op)

    def visit_sub(self, op):
        self._visit_binary_op(op)

    def visit_mul(self, op):
        self._visit_binary_op(op)

    def visit_div(self, op):
        self._visit_binary_op(op)

    def visit_mod(self, op):
        self._visit_binary_op(op)

    def visit_floor_div(self, op):
        self._visit_binary_op(op)

    def visit_floor_mod(self, op):
        self._visit_binary_op(op)

    def visit_min(self, op):
        self._visit_binary_op(op)

    def visit_max(self, op):
        self._visit_binary_op(op)

    def visit_eq(self, op):
        self._visit_binary_op(op)

    def visit_ne(self, op):
        self._visit_binary_op(op)

    def visit_lt(self, op):
        self._visit_binary_op(op)

    def visit_le(self, op):
        self._visit_binary_op(op)

    def visit_gt(self, op):
        self._visit_binary_op(op)

    def visit_ge(self, op):
        self._visit_binary_op(op)

    def visit_and(self, op):
        self._visit_binary_op(op)

    def visit_or(self, op):
        self._visit_binary_op(op)

    def visit_reduce(self, reduce_op):
        for iv in reduce_op.axis:
            self.visit_expr(iv.dom.min)
            self.visit_expr(iv.dom.extent)

        for src in reduce_op.source:
            self.visit_expr(src)

        for init in reduce_op.init:
            self.visit_expr(init)

        self.visit_expr(reduce_op.condition)

    def visit_cast(self, cast):
        self.visit_expr(cast.value)

    def visit_not(self, not_op):
        self.visit_expr(not_op.a)

    def visit_select(self, select):
        self.visit_expr(select.condition)
        self.visit_expr(select.true_value)
        self.visit_expr(select.false_value)

    def visit_ramp(self, ramp):
        self.visit_expr(ramp.base)
        self.visit_expr(ramp.stride)

    def visit_broadcast(self, broadcast):
        self.visit_expr(broadcast.value)

    def visit_shuffle(self, shuffle):
        for idx in shuffle.indices:
            self.visit_expr(idx)

        for vector in shuffle.vectors:
            self.visit_expr(vector)

    def visit_int_imm(self, _):
        pass

    def visit_float_imm(self, _):
        pass

    def visit_string_imm(self, _):
        pass

    def visit_any(self, _):
        pass


class ExprMutator(ExprFunctor):
    """A mutator over TIR expression.

    The default behavior recursively traverses and reconstructs the TIR expression.
    """

    def visit_var(self, var):
        return var

    def visit_size_var(self, size_var):
        return self.visit_var(size_var)

    def visit_buffer_load(self, buf_load):
        new_indices = [self.visit_expr(idx) for idx in buf_load.indices]
        new_pred = self.visit_expr(buf_load.predicate)
        if new_indices == list(buf_load.indices) and new_pred == buf_load.predicate:
            return buf_load
        return _expr.BufferLoad(buf_load.buffer, new_indices, new_pred, buf_load.span)

    def visit_producer_load(self, producer_load):
        new_indices = [self.visit_expr(idx) for idx in producer_load.indices]
        if new_indices == list(producer_load.indices):
            return producer_load
        return _expr.ProducerLoad(producer_load.producer, new_indices, producer_load.span)

    def visit_let(self, let):
        new_value = self.visit_expr(let.value)
        new_body = self.visit_expr(let.body)
        if new_value == let.value and new_body == let.body:
            return let
        return _expr.Let(let.var, new_value, new_body, let.span)

    def visit_call(self, call):
        new_args = [self.visit_expr(arg) for arg in call.args]
        if new_args == list(call.args):
            return call
        return _expr.Call(call.dtype, call.op, new_args, call.span)

    def _visit_binary_op(self, bin_op):
        new_a = self.visit_expr(bin_op.a)
        new_b = self.visit_expr(bin_op.b)
        if new_a == bin_op.a and new_b == bin_op.b:
            return bin_op
        return bin_op.__class__(new_a, new_b, bin_op.span)

    def visit_add(self, op):
        return self._visit_binary_op(op)

    def visit_sub(self, op):
        return self._visit_binary_op(op)

    def visit_mul(self, op):
        return self._visit_binary_op(op)

    def visit_div(self, op):
        return self._visit_binary_op(op)

    def visit_mod(self, op):
        return self._visit_binary_op(op)

    def visit_floor_div(self, op):
        return self._visit_binary_op(op)

    def visit_floor_mod(self, op):
        return self._visit_binary_op(op)

    def visit_min(self, op):
        return self._visit_binary_op(op)

    def visit_max(self, op):
        return self._visit_binary_op(op)

    def visit_eq(self, op):
        return self._visit_binary_op(op)

    def visit_ne(self, op):
        return self._visit_binary_op(op)

    def visit_lt(self, op):
        return self._visit_binary_op(op)

    def visit_le(self, op):
        return self._visit_binary_op(op)

    def visit_gt(self, op):
        return self._visit_binary_op(op)

    def visit_ge(self, op):
        return self._visit_binary_op(op)

    def visit_and(self, op):
        return self._visit_binary_op(op)

    def visit_or(self, op):
        return self._visit_binary_op(op)

    def _visit_iter_var(self, iv):
        dom = iv.dom
        new_min = self.visit_expr(dom.min)
        new_extent = self.visit_expr(dom.extent)
        if new_min == dom.min and new_extent == dom.extent:
            return iv
        new_dom = ir.Range.from_min_extent(new_min, new_extent, dom.span)
        return _expr.IterVar(new_dom, iv.var, iv.iter_type, iv.thread_tag, iv.span)

    def visit_reduce(self, reduce_op):
        new_axis = [self._visit_iter_var(iv) for iv in reduce_op.axis]
        new_source = [self.visit_expr(src) for src in reduce_op.source]
        new_init = [self.visit_expr(x) for x in reduce_op.init]
        new_condition = self.visit_expr(reduce_op.condition)

        if (
            new_axis == list(reduce_op.axis)
            and new_source == list(reduce_op.source)
            and new_init == list(reduce_op.init)
            and new_condition == reduce_op.condition
        ):
            return reduce_op
        return _expr.Reduce(
            reduce_op.combiner,
            new_source,
            new_axis,
            new_condition,
            reduce_op.value_index,
            new_init,
            reduce_op.span,
        )

    def visit_cast(self, cast):
        new_value = self.visit_expr(cast.value)
        if new_value == cast.value:
            return cast
        return _expr.Cast(cast.dtype, new_value, cast.span)

    def visit_not(self, not_op):
        new_a = self.visit_expr(not_op.a)
        if new_a == not_op.a:
            return not_op
        return _expr.Not(new_a, not_op.span)

    def visit_select(self, select):
        new_condition = self.visit_expr(select.condition)
        new_true_value = self.visit_expr(select.true_value)
        new_false_value = self.visit_expr(select.false_value)
        if (
            new_condition == select.condition
            and new_true_value == select.true_value
            and new_false_value == select.false_value
        ):
            return select
        return _expr.Select(new_condition, new_true_value, new_false_value, select.span)

    def visit_ramp(self, ramp):
        new_base = self.visit_expr(ramp.base)
        new_stride = self.visit_expr(ramp.stride)
        if new_base == ramp.base and new_stride == ramp.stride:
            return ramp
        return _expr.Ramp(new_base, new_stride, ramp.lanes, ramp.span)

    def visit_broadcast(self, broadcast):
        new_value = self.visit_expr(broadcast.value)
        if new_value == broadcast.value:
            return broadcast
        return _expr.Broadcast(new_value, broadcast.lanes, broadcast.span)

    def visit_shuffle(self, shuffle):
        new_vectors = [self.visit_expr(vector) for vector in shuffle.vectors]
        if new_vectors == list(shuffle.vectors):
            return shuffle
        return _expr.Shuffle(new_vectors, shuffle.indices, shuffle.span)

    def visit_int_imm(self, op):
        return op

    def visit_float_imm(self, op):
        return op

    def visit_string_imm(self, op):
        return op

    def visit_any(self, op):
        return op
