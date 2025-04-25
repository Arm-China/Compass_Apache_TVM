# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=too-many-nested-blocks
"""Revert 'transpose' for fuse_pass."""
from tvm import relax, ir
from tvm.relax.dpl import wildcard, is_op, rewrite_call, is_const


class ExchangeDequantPerm:
    """permute_dims(deq(inp)) --> deq(permute_dims(inp))"""

    def __init__(self):
        self.inp = wildcard()
        self.dequant = is_op("relax.dequantize")(self.inp, is_const(), is_const())
        self.permute_dims = is_op("relax.permute_dims")(self.dequant)
        self.pattern = self.permute_dims

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):  # pylint: disable=unused-argument
            inp = matches[self.inp]
            dequant = matches[self.dequant]
            permute_dims = matches[self.permute_dims]
            new_perm = relax.Call(permute_dims.op, [inp], permute_dims.attrs)
            new_dequant = relax.Call(dequant.op, [new_perm] + dequant.args[1:], dequant.attrs)
            return new_dequant

        return self.pattern, rewriter


@ir.transform.module_pass(opt_level=0)
class RevertTranspose:
    """Revert 'transpose' for fuse_pass."""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        ir_mod["main"] = rewrite_call(*ExchangeDequantPerm().pr, ir_mod["main"])

        return ir_mod
