# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Rewrite function by pattern"""
from tvm.relax import transform
from tvm.relax.dpl import rewrite_call
from .pattern_rewrites import UpdateMatmul, ReorderMatmulReshapeAdd, MergeAdjacentReshape
from .pattern_rewrites import EliminateUselessPermuteDims, MergeQuantCast, MergePermMean
from .pattern_rewrites import EliminateIdentityOp, AddToMul, FoldDilatedConv2d, Conv1DToConv2D
from .pattern_rewrites import ReorderConv2dReshapeAddActivation, SimplifyConsecutivePermuteDims
from .pattern_rewrites import ReorderBinaryOpsConstArgs, BroadcastToTile, MergeAddSubToSub
from .pattern_rewrites import AdjustArgMinMaxKeepDim, SubToMulADD


@transform.function_pass(opt_level=0)
class PatternRewriteInOptimize:
    """Rewrite function by pattern"""

    def transform_function(self, func, mod, pass_ctx):  # pylint: disable=unused-argument
        """Transform the given function and return the result."""
        opts = (
            ReorderBinaryOpsConstArgs(),
            MergeAddSubToSub(),
            UpdateMatmul(),
            BroadcastToTile(),
            ReorderMatmulReshapeAdd(),
            MergeAdjacentReshape(),
            EliminateUselessPermuteDims(),
            MergeQuantCast(),
            MergePermMean(),
            EliminateIdentityOp(),
            AdjustArgMinMaxKeepDim(),
            AddToMul(),
            Conv1DToConv2D(),
            ReorderConv2dReshapeAddActivation(),
            FoldDilatedConv2d(),
            EliminateIdentityOp(),
            SimplifyConsecutivePermuteDims(),
            SubToMulADD(),
        )

        updated_func = func
        for opt in opts:
            updated_func = rewrite_call(*opt.pr, updated_func)

        return updated_func
