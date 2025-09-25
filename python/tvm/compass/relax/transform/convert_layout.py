# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Layout conversion on specify function."""
from tvm import relax, ir


@ir.transform.module_pass(opt_level=0)
class ConvertLayout:
    """Layout conversion on specify function."""

    def __init__(self, func_name=None, desired_layouts=None):
        self.func_name = func_name
        self.desire_layouts = desired_layouts

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        if not self.func_name or not self.desire_layouts:
            return ir_mod
        if self.func_name not in [x.name_hint for x in ir_mod.get_global_vars()]:
            return ir_mod
        update_mod = relax.transform.ConvertLayout(self.desire_layouts)(ir_mod)
        if len(ir_mod.get_global_vars()) == 1:
            return update_mod

        for gvar in ir_mod.get_global_vars():
            if gvar.name_hint == self.func_name:
                for gvar1 in update_mod.get_global_vars():
                    if gvar1.name_hint == self.func_name:
                        ir_mod[gvar] = update_mod[gvar1]

        return ir_mod
