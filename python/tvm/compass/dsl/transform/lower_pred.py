# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Lower the standard TIR predicate nodes to the corresponding representation of Compass."""
from tvm import ir, tir


def vmov(x, lanes):
    lanes = lanes.value if isinstance(lanes, tir.IntImm) else lanes
    assert lanes in (8, 16, 32), f"Expected lanes to be 8/16/32, but got lanes={lanes}."
    return tir.call_extern(f"boolx{lanes}", f'__vmov_{("w", "h", "b")[lanes // 16]}', x)


class _Mutator(tir.StmtExprMutator):
    # Zhouyi NPU's predicate value is 32-bit, for operand whose element is "word"(e.g., int32x8) the
    # last bit of every 4 bits valid(e.g., 0bxxx1: enable, 0bxxx0: disable), for "half" operand
    # (e.g., int16x16) the last bit of every 2 bits valid(e.g., 0bx1: enable, 0bx0: disable), for
    # "byte" operand(e.g., int8x32) every bit valid(e.g., 0b1: enable, 0b0: disable).

    def _mutate_const_pred(self, call):
        # Convert the boolean array to the predicate representation of Compass.
        lanes = call.dtype.lanes
        ret = 0
        for i, x in enumerate(call.args):
            if bool(x):
                ret |= 1 << (i * (32 // lanes))
        return vmov(ret, lanes)

    def _mutate_low_true_pred(self, call):
        n, lanes = call.args[0], call.dtype.lanes
        # Here lanes is a compile time const, so no need to optmize 32 // lanes.
        # When 32 // lanes only can be 1, 2, 4, n * (32 // lanes) equals n << ((32 // lanes) >> 1).
        true_count = n << ((32 // lanes) >> 1)
        # If true_count in [0, 31], (1 << true_count) - 1
        # If true_count equals 32, (0 << 32) - 1
        return vmov((((true_count >> 5) ^ 1) << true_count) - 1, lanes)

    def visit_call(self, call):
        ret = super().visit_call(call)

        if ret.op == ir.Op.get("tir.const_pred"):
            return self._mutate_const_pred(ret)
        if ret.op == ir.Op.get("tir.low_true_pred"):
            return self._mutate_low_true_pred(ret)

        return ret


@tir.transform.prim_func_pass(opt_level=0)
class LowerPred:
    """Lower the standard TIR predicate nodes to the corresponding representation of Compass.

    Precondition
      - All constant predicate nodes must be represented by "tir.const_pred", need to be guaranteed
        by script APIs and pass "LowerStandard".
      - The length of tail or constant predicate nodes must equal to hardware vector width, need to
        be guaranteed by pass "AlignVectorWidthBySplit" and "AlignVectorWidthByPad".
    """

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        return func.with_body(_Mutator().visit(func.body), span=func.span)
