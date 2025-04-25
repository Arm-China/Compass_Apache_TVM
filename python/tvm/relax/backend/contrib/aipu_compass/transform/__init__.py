# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The Relay IR namespace contains AIPU Compass extended transform passes."""
from .transform import FuseTuple
from .build_aipu_subgraph import BuildAipuSubgraph
from .rename_aipu_subfunc import RenameAipuSubfunc
from .hint_pattern_rewrite import HintPatternRewrite
from .convert_ops import ConvertOps
from .sink_transpose import SinkTranspose
from .revert_transpose import RevertTranspose
from .sink_dequantize import SinkDequantize
from .copy_dequantize import CopyDequantize
from .save_inp_quant_info import SaveInpQuantInfo
