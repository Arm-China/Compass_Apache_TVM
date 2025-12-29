# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Ensure a prim func is well formed."""
from tvm import tir, ir


class _Analyzer(tir.StmtExprVisitor):
    def __init__(self):
        super().__init__()
        self._bads = set()

    def _check_vload(self, call):
        dtype = call.dtype
        mask_dtype = call.args[3].dtype
        if dtype.lanes != mask_dtype.lanes:
            message = "========== vload: lanes != mask.lanes ==========\n"
            message += str(call)
            self._bads.add(message)

    def _check_vstore(self, call):
        dtype = call.args[1].dtype
        mask_dtype = call.args[4].dtype
        if dtype.lanes != mask_dtype.lanes:
            message = "========== vstore: value.lanes != mask.lanes ==========\n"
            message += str(call) + "\n"
            self._bads.add(message)

    def visit_call(self, call):
        super().visit_call(call)

        if call.op != ir.Op.get("tir.call_extern"):
            return

        func_name = call.args[0].value
        if func_name == "vload":
            self._check_vload(call)
        elif func_name == "vstore":
            self._check_vstore(call)

    def check(self, func):
        self.visit(func.body)
        if len(self._bads) != 0:
            message = "\n"
            for bad in self._bads:
                message += bad
                message += "-" * 100 + "\n"
            raise RuntimeError(f"The prim func isn't well formed with bads as follows: {message}")


def ensure_well_formed(prim_func):
    """Ensure a prim func is well formed.

    Check items:
    1. vload:  dtype.lanes == mask_dtype.lanes
    2. vstore: value_dtype.lanes == mask_dtype.lanes
    """
    return _Analyzer().check(prim_func)
