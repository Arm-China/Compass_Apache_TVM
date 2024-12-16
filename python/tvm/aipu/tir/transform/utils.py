# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Common TIR transform utilities."""
from tvm import ir, tir
from tvm.arith import detect_linear_equation
from tvm.ir.base import structural_equal
from . import _ffi_api


def is_all_true_pred(pred):
    return _ffi_api.is_all_true_pred(pred)


def is_builtin(x, names):
    if not (isinstance(x, tir.Call) and x.op == ir.Op.get("tir.call_extern")):
        return False

    names = (names,) if isinstance(names, str) else names
    return any(x.args[0] == name for name in names)


def is_broadcast_const(x):
    return is_builtin(x, "__vbcast") and isinstance(x.args[2], (tir.IntImm, tir.FloatImm))


def is_const_pred(x):
    return isinstance(x, tir.Call) and x.op == ir.Op.get("tir.const_pred")


def is_low_true_pred(x):
    return isinstance(x, tir.Call) and x.op == ir.Op.get("tir.low_true_pred")


def is_pointer(x):
    return isinstance(x, tir.Call) and x.op == ir.Op.get("tir.pointer")


class _NormalizeComparisons(tir.ExprMutator):
    """Normalize comparisons to less."""

    def __init__(self, var):
        super().__init__()
        self.var = var

    def visit_lt(self, op):
        m = detect_linear_equation(op.a, [self.var])
        if len(m) == 0 or m[0].value != 1:
            return op
        return tir.LT(self.var, op.b - m[1])

    def visit_le(self, op):
        m = detect_linear_equation(op.a, [self.var])
        if len(m) == 0 or m[0].value != 1:
            return op
        return tir.Lt(self.var, op.b - m[1] - 1)


def get_normalized_condition(var, origin_cond):
    normalize_cond = _NormalizeComparisons(var).visit_expr(origin_cond)

    if structural_equal(normalize_cond, origin_cond):
        return None
    return normalize_cond.b
