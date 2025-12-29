# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Rewrite function by pattern"""
from tvm.relax import transform
from tvm.relax.dpl import rewrite_call
from .pattern_rewrites import MergeRehapeTransReshape, NormalizeSqueezeAxis


@transform.function_pass(opt_level=0)
class PatternRewriteBeforeConvertLayout:
    """Rewrite function by pattern"""

    def transform_function(self, func, mod, pass_ctx):  # pylint: disable=unused-argument
        """Transform the given function and return the result."""
        opts = (
            NormalizeSqueezeAxis(),
            MergeRehapeTransReshape(),
        )

        updated_func = func
        for opt in opts:
            updated_func = rewrite_call(*opt.pr, updated_func)

        return updated_func
