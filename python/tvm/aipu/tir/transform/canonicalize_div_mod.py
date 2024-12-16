# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""
Canonicalize the FloorDiv to TruncDiv and FloorMod to TruncMod.
And optimize if the right operand is integer power of 2.
"""
import math
from tvm import tir


class _Canonicalizer(tir.StmtExprMutator):
    def _is_const_power_of_two_integer(self, op):
        if isinstance(op, tir.IntImm):
            n = op.value
            return n > 0 and (n & (n - 1)) == 0
        return False

    def visit_floor_div(self, op):
        ret = super().visit_floor_div(op)
        if self._is_const_power_of_two_integer(ret.b):
            return tir.shift_right(ret.a, int(math.log2(ret.b.value)))
        return tir.truncdiv(ret.a, ret.b, ret.span)

    def visit_floor_mod(self, op):
        ret = super().visit_floor_mod(op)
        if self._is_const_power_of_two_integer(ret.b):
            return tir.bitwise_and(ret.a, ret.b.value - 1)
        return tir.truncmod(ret.a, ret.b, ret.span)


@tir.transform.prim_func_pass(opt_level=0)
class CanonicalizeDivMod:
    """
    Canonicalize the FloorDiv to TruncDiv and FloorMod to TruncMod.
    And optimize if the right operand is integer power of 2.
    """

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        return func.with_body(_Canonicalizer().visit(func.body), span=func.span)
