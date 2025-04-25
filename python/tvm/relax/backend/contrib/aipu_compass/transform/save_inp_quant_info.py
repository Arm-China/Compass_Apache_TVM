# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=too-many-nested-blocks
"""Save inp quant infos to attrs of aipu subfunc."""
from tvm import relax, ir
from tvm.relax.expr_functor import PyExprVisitor, visitor
from .utils import is_call
from ..config import AipuCompassConfig


@visitor
class Visitor(PyExprVisitor):
    """Get inp quant infos of aipu subfunc."""

    def __init__(self, var2val):
        super().__init__()
        self.var2val = var2val
        self.gv2qinfos = {}

    def visit_call_(self, call):
        if not isinstance(call.op, relax.GlobalVar):
            return super().visit_call_(call)
        if not call.op.name_hint.startswith("tvm_aipu_compass"):
            return super().visit_call_(call)

        infos = []
        for arg in call.args:
            quant = self.var2val.get(arg, arg)
            if is_call(quant, "quantize"):
                infos.append(quant.args[1:])
        self.gv2qinfos[call.op] = infos

        return super().visit_call_(call)

    def get_gv2qinfos(self, func):
        self.visit_expr(func)
        return self.gv2qinfos


@ir.transform.module_pass(opt_level=0)
class SaveInpQuantInfo:
    """Save inp quant infos to attrs of aipu subfunc."""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        if AipuCompassConfig.get().common["compat_quantized_model"] != "true":
            return ir_mod

        main_func = ir_mod["main"]
        var2val = relax.analysis.get_var2val(main_func)
        gv2qinfos = Visitor(var2val).get_gv2qinfos(main_func)
        for gvar, func in ir_mod.functions.items():
            if gvar not in gv2qinfos:
                continue
            ir_mod[gvar] = func.with_attr("quant_infos", gv2qinfos[gvar])

        return ir_mod
