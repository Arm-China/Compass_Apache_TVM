# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Insert the event state initialization statement when "S.alloc_events" have been used."""
from tvm import tir, ir
from .utils import is_builtin


class _Analyzer(tir.StmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.contain_alloc_events = False

    def visit_call(self, call):
        super().visit_call(call)

        if is_builtin(call, "alloc_event"):
            self.contain_alloc_events = True


@ir.transform.module_pass(opt_level=0)
class InitializeEventState:
    """Insert the event state initialization statement when "S.alloc_events" have been used."""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        analyzer = _Analyzer()
        entry_gvar, entry_func = None, None

        for gvar, func in ir_mod.functions.items():
            if func.attrs["tir.is_entry_func"]:
                entry_gvar, entry_func = gvar, func
            analyzer.visit(func.body)

        if not analyzer.contain_alloc_events:
            return ir_mod

        event_state_init = tir.Evaluate(tir.precodegen("*addr_of_event_state() = 0"))
        new_body = tir.SeqStmt((event_state_init, entry_func.body))
        ir_mod[entry_gvar] = entry_func.with_body(new_body, span=entry_func.span)
        return ir_mod
