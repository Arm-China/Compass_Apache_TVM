# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The Relax IR namespace contains Zhouyi Compass extended transform passes."""
from .transform import FuseTuple
from .build_compass_subgraph import BuildCompassSubgraph
from .rename_compass_subfunc import RenameCompassSubfunc
from .pattern_rewrite_in_optimize import PatternRewriteInOptimize
from .pattern_rewrite_before_convert_layout import PatternRewriteBeforeConvertLayout
from .pattern_rewrite_after_convert_layout import PatternRewriteAfterConvertLayout
from .pattern_rewrite_after_fuse_ops import PatternRewriteAfterFuseOps
from .convert_compass_ops import ConvertCompassOps
from .sink_transpose import SinkTranspose
from .revert_transpose import RevertTranspose
from .sink_dequantize import SinkDequantize
from .copy_dequantize import CopyDequantize
from .save_inp_quant_info import SaveInpQuantInfo
from .find_postprocess import GetPostProcessFunction
from .inline_single_op_func import InlineSingleOpFunc
from .convert_layout import ConvertLayout
from .rewrite_ops_in_optimize import RewriteOpsInOptimize
from .pattern_rewrite_after_partition import PatternRewriteAfterPartition
from .legalize_var_name import LegalizeVarName
from .rearrange_params import RearrangeParams
from .prune_compass_subgraph import PruneCompassSubGraphs
from .unique_var_name import UniqueVarName
from .fallback_nodes_to_cpu import FallbackNodesToCPU
from .extract_in_out_quant_ops import ExtractInOutQuantOps
