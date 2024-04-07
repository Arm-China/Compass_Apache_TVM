# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The AIPU Compass extended Relay analysis passes."""
from tvm import ir, relay, IRModule
from .....analysis import _ffi_api


def has_quantized_op(ir_mod):
    """Check whether the given IRModule contains quantized operator or not."""
    qnn_ops = []

    def _visit_expr(expr):
        if (
            isinstance(expr, relay.Call)
            and isinstance(expr.op, ir.Op)
            and expr.op.name.startswith("qnn.")
        ):
            qnn_ops.append(expr)

    relay.analysis.post_order_visit(ir_mod["main"], _visit_expr)

    return len(qnn_ops) != 0


def printer_index_to_expr(mod):
    """Extract Relay Expr by printer SSA ID

    This function is used for extracting Relay Expr
    by printer SSA ID of the main function
    that we can see in `print(mod["main"])`.

    Parameters
    ----------
    mod : tvm.IRModule or Relay Function

    Returns
    -------
    ret : Dict[int, tvm.relay.function.Function]

    Examples
    --------
    .. code-block:: python

        # Suppose our module is printed like this:
        # def @main(%x: Tensor[(1, 1, 5, 1), float32], %w1, %w2) {
        #   %0 = nn.conv2d(%x, %w1, padding=[1, 1, 1, 1], channels=1, kernel_size=[3, 3]);
        #   %1 = nn.conv2d(%0, %w2, padding=[1, 1, 1, 1], channels=1, kernel_size=[3, 3]);
        #   %2 = add(%0, %1);
        #   %3 = split(%2, indices_or_sections=1);
        #   %4 = %3.0;
        #   add(%4, 1f)
        # }
        # if we want to extract `%1 = nn.conv2d`
        from tvm import relay

        relay.analysis.printer_index_to_expr(mod)[1]
    """
    if not isinstance(mod, (IRModule, relay.Function)):
        raise RuntimeError("printer_index_to_expr only support input as IRModule or relay.Function")
    func = mod["main"] if isinstance(mod, IRModule) else mod
    memo = dict()
    for idx, expr in enumerate(_ffi_api.PrinterIndexToExpr(func)):
        memo[idx] = expr
    memo[len(memo)] = func.body
    for var in mod["main"].params:
        memo[var.name_hint] = var
    return memo
