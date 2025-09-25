# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The Zhouyi Compass extended Relax analysis passes."""
from tvm import ir, relax


def has_quantized_op(ir_mod):
    """Check whether the given IRModule contains quantize/dequantize operator or not."""
    qdq_ops = []

    def _visit_expr(expr):
        if (
            isinstance(expr, relax.Call)
            and isinstance(expr.op, ir.Op)
            and expr.op.name in ["relax.quantize", "relax.dequantize"]
        ):
            qdq_ops.append(expr)

    relax.analysis.post_order_visit(ir_mod["main"], _visit_expr)

    return len(qdq_ops) > 0
