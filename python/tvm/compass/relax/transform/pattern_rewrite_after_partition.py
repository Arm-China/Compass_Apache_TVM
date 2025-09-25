# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Pattern rewrite after partition."""
from tvm.relax import transform
from tvm.relax.dpl import rewrite_call
from .pattern_rewrites import UnBindDequantQuant
from .utils import is_compass_func, is_cps_composite_func


@transform.function_pass(opt_level=0)
class PatternRewriteAfterPartition:
    """Rewrite function by pattern after partition"""

    def __init__(self, except_cps_func=True):
        self.is_except_cps_func = except_cps_func

    def transform_function(self, func, mod, pass_ctx):  # pylint: disable=unused-argument
        """Transform the given function and return the result."""
        if self.is_except_cps_func and (is_compass_func(func) or is_cps_composite_func(func)):
            return func
        opts = (UnBindDequantQuant(),)

        updated_func = func
        for opt in opts:
            updated_func = rewrite_call(*opt.pr, func)

        return updated_func
