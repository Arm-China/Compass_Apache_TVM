# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Rewrite func except compass func by pattern after fused ops"""
from tvm.relax import transform
from tvm.relax.dpl import rewrite_call
from .pattern_rewrites import SimplifyAddZeroMulOne
from .utils import is_compass_func, is_cps_composite_func


@transform.function_pass(opt_level=0)
class PatternRewriteAfterFuseOps:
    """Rewrite func except compass func by pattern after fused ops"""

    def transform_function(self, func, mod, pass_ctx):  # pylint: disable=unused-argument
        """Transform the given function and return the result."""
        if is_compass_func(func) or is_cps_composite_func(func):
            return func
        opts = (SimplifyAddZeroMulOne(),)

        updated_func = func
        for opt in opts:
            updated_func = rewrite_call(*opt.pr, updated_func)

        return updated_func
