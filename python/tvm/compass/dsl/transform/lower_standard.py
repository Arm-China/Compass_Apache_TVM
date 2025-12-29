# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Lower the standard TIR nodes to the corresponding representation of Compass."""
from tvm import ir, tir
from .. import script as S


class _Lower(tir.StmtExprMutator):
    def visit_buffer_load(self, buf_load):
        ret = super().visit_buffer_load(buf_load)
        dtype = ret.dtype
        if dtype.is_scalar:
            return ret

        stride = 1
        idx = ret.indices[0]
        if isinstance(idx, tir.Ramp):
            stride = idx.stride
            idx = idx.base

        pred = ret.predicate
        if pred is None:
            pred = [True] * dtype.lanes
        return S.vload(ret.buffer.addr_of(idx), pred, dtype.lanes, stride)

    def visit_buffer_store(self, buf_store):
        ret = super().visit_buffer_store(buf_store)
        val_dtype = ret.value.dtype
        if val_dtype.is_scalar:
            return ret

        stride = 1
        idx = ret.indices[0]
        if isinstance(idx, tir.Ramp):
            stride = idx.stride
            idx = idx.base

        pred = ret.predicate
        if pred is None:
            pred = [True] * val_dtype.lanes
        vstore = S.vstore(ret.value, ret.buffer.addr_of(idx), pred, stride)
        return tir.Evaluate(vstore, ret.span)

    def visit_add(self, op):
        ret = super().visit_add(op)
        return S.vadd(ret.a, ret.b) if ret.dtype.is_vector else ret

    def visit_sub(self, op):
        ret = super().visit_sub(op)
        return S.vsub(ret.a, ret.b) if ret.dtype.is_vector else ret

    def visit_mul(self, op):
        ret = super().visit_mul(op)
        return S.vmul(ret.a, ret.b) if ret.dtype.is_vector else ret

    def visit_div(self, op):
        ret = super().visit_div(op)
        # The integer vector division instruction will get wrong result, when generated as "x / y"
        # where x=-128, y=-1(constant value).
        return S.vdiv(ret.a, ret.b) if ret.dtype.is_vector else ret

    def visit_lt(self, op):
        ret = super().visit_lt(op)
        return S.vclt(ret.a, ret.b) if ret.dtype.is_vector else ret

    def visit_gt(self, op):
        ret = super().visit_gt(op)
        return S.vcgt(ret.a, ret.b) if ret.dtype.is_vector else ret

    def visit_le(self, op):
        ret = super().visit_le(op)
        return S.vcle(ret.a, ret.b) if ret.dtype.is_vector else ret

    def visit_ge(self, op):
        ret = super().visit_ge(op)
        return S.vcge(ret.a, ret.b) if ret.dtype.is_vector else ret

    def visit_eq(self, op):
        ret = super().visit_eq(op)
        return S.vceq(ret.a, ret.b) if ret.dtype.is_vector else ret

    def visit_ne(self, op):
        ret = super().visit_ne(op)
        return S.vcneq(ret.a, ret.b) if ret.dtype.is_vector else ret

    def visit_broadcast(self, broadcast):
        ret = super().visit_broadcast(broadcast)
        assert isinstance(ret.lanes, tir.IntImm), f"Unexpected type: {type(ret.lanes)}."
        return S.vbcast(ret.value, lanes=ret.lanes.value)

    def visit_min(self, op):
        ret = super().visit_min(op)
        return S.min(ret.a, ret.b) if ret.dtype.is_vector else ret

    def visit_max(self, op):
        ret = super().visit_max(op)
        return S.max(ret.a, ret.b) if ret.dtype.is_vector else ret

    def visit_cast(self, cast):
        ret = super().visit_cast(cast)
        return S.cast(ret.value, ret.dtype) if ret.dtype.is_vector else ret

    def visit_call(self, call):
        ret = super().visit_call(call)

        if ret.op == ir.Op.get("tir.shift_left"):
            return S.vsl(ret.args[0], ret.args[1]) if ret.dtype.is_vector else ret
        if ret.op == ir.Op.get("tir.shift_right"):
            return S.vsr(ret.args[0], ret.args[1]) if ret.dtype.is_vector else ret
        if ret.op == ir.Op.get("tir.bitwise_and"):
            return S.vand(ret.args[0], ret.args[1]) if ret.dtype.is_vector else ret
        if ret.op == ir.Op.get("tir.bitwise_or"):
            return S.vor(ret.args[0], ret.args[1]) if ret.dtype.is_vector else ret
        if ret.op == ir.Op.get("tir.bitwise_xor"):
            return S.vxor(ret.args[0], ret.args[1]) if ret.dtype.is_vector else ret
        if ret.op == ir.Op.get("tir.pow"):
            return S.pow(ret.args[0], ret.args[1])
        return ret

    def visit_select(self, select):
        ret = super().visit_select(select)
        if ret.dtype.is_scalar:
            return ret
        return S.vsel(ret.true_value, ret.false_value, ret.condition)


@tir.transform.prim_func_pass(opt_level=0)
class LowerStandard:
    """Lower the standard TIR nodes to the corresponding representation of Compass.

    After this pass, all vector load/store nodes are represented by "vload"/"vstore",
    "tir.BufferLoad"/"tir.BufferStore" only responsible for representing the scalar load/store
    nodes, all vector compare/broadcast/min/max expr are represented by Zhouyi NPU compare
    instruction.
    """

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        return func.with_body(_Lower().visit(func.body), span=func.span)
