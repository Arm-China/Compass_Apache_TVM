# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Rewrite function by pattern"""
from tvm.relax import transform
from tvm.relax.dpl import rewrite_call
from .pattern_rewrites import MergeConstToConvWeight, MergeMultiplyToConvWeight, MergeExplicitPad
from .pattern_rewrites import EliminateUselessPermuteDims, MergeAdjacentReshape, MergePermMean
from .pattern_rewrites import MergeConvAddMulToConvAdd, ConvertToReshape, MergeConstToFcWeight
from .pattern_rewrites import SimplifyTransReshapeTrans, RevertReshape, BindDequantQuant
from .pattern_rewrites import EliminateIdentityOp, SimplifyConsecutivePermuteDims
from .pattern_rewrites import ConvertMeanToPool, AdjustQnnMeanKeepDim, ExtractNegativePadFromConv2d


@transform.function_pass(opt_level=0)
class PatternRewriteAfterConvertLayout:
    """Rewrite function by pattern"""

    def transform_function(self, func, mod, pass_ctx):  # pylint: disable=unused-argument
        """Transform the given function and return the result."""
        opts = (
            ExtractNegativePadFromConv2d(),
            MergeConstToConvWeight(),
            MergeConvAddMulToConvAdd(),
            MergeMultiplyToConvWeight(),
            MergeConstToFcWeight(),
            MergeExplicitPad(),
            EliminateUselessPermuteDims(),
            ConvertToReshape(),
            SimplifyTransReshapeTrans(),
            AdjustQnnMeanKeepDim(),
            MergeAdjacentReshape(),
            RevertReshape(),
            MergeAdjacentReshape(),
            EliminateIdentityOp(),
            MergePermMean(),
            ConvertMeanToPool(),
            BindDequantQuant(),
            SimplifyConsecutivePermuteDims(),
        )

        updated_func = func
        for opt in opts:
            updated_func = rewrite_call(*opt.pr, updated_func)

        return updated_func
