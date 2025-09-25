# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Fallback Compass subgraph nodes to CPU."""
from tvm import relax, ir


@ir.transform.module_pass(opt_level=0)
class FallbackNodesToCPU:
    """Fallback Compass subgraph nodes to CPU."""

    def __init__(self, fallback_nodes):
        self.fallback_nodes = fallback_nodes

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        if self.fallback_nodes is None:
            return ir_mod

        global_vars_to_fallback = []
        for name, node_names in self.fallback_nodes.items():
            if not any(name == x.name_hint for x in ir_mod.get_global_vars()):
                continue
            caller_func = ir_mod[name]
            assert len(caller_func.body.blocks) == 1, "Unsupported multi blocks now."
            bindings = caller_func.body.blocks[0].bindings
            inline_dict = {}
            for node_name in node_names:
                for bind in bindings:
                    if (
                        bind.var.name_hint == node_name
                        and isinstance(bind.value, relax.Call)
                        and isinstance(bind.value.op, ir.GlobalVar)
                        and "Composite" in ir_mod[bind.value.op].attrs
                        and ir_mod[bind.value.op].attrs["Composite"].startswith("compass")
                    ):
                        fb_gvar = bind.value.op
                        inline_dict[fb_gvar.name_hint] = ir_mod[fb_gvar]
                        global_vars_to_fallback.append(fb_gvar)
            if len(inline_dict) == 0:
                continue
            ir_mod[name] = caller_func.inline_functions(inline_dict)

        # Delete all compass subfunc to be fall back.
        for gvar in global_vars_to_fallback:
            del ir_mod[gvar]

        return ir_mod
