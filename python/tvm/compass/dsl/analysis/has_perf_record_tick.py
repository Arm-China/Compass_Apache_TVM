# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Check whether contain perf record tick or not."""
from tvm import tir, ir


class _Analyzer(tir.StmtExprVisitor):
    def __init__(self):
        super().__init__()
        self._has_perf_tick = False

    def visit_call(self, call):
        super().visit_call(call)

        if call.op != ir.Op.get("tir.call_extern"):
            return

        name = call.args[0]
        if name in ("__perf_record_tick_begin", "__perf_record_tick_end"):
            self._has_perf_tick = True

    def check(self, func):
        self.visit(func.body)
        return self._has_perf_tick


def has_perf_record_tick(ir_mod):
    """Check whether contain perf record tick or not."""
    return any(_Analyzer().check(x) for x in ir_mod.functions.values())
