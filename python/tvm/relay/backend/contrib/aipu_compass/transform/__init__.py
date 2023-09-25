# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""The Relay IR namespace contains AIPU Compass extended transform passes."""
from .transform import *
from .simplify_pad import SimplifyPad
from .sink_transpose import SinkTranspose
from .evaluate_zero_free_args_call import EvaluateZeroFreeArgsCall
from .convert_aipu_ops import ConvertAIPUOps
from .find_aipu_postprocess import GetPostProcessFunction
from .hint_pattern_rewrite import HintPatternRewrite
from .extract_negative_pad_from_conv2d import ExtractNegativePadFromConv2d
from .merge_loop_structure import LoopStructureMerger
from .legailize_var_name import LegalizeVarName
from .unroll_let_loop_in_main import UnrollLetLoopInMain
from .verify_let_loop_params import VerifyLetLoopParams
from .eliminate_tensor_array_ops import EliminateTensorArrayOp
from .prune_aipu_subgraphs import PruneAIPUSubGraphs
from .split_deformable_conv2d import SplitDeformableConv2d
from .rearrange_params import RearrangeParams
from .rearrange_names import RearrangeNames
from .build_aipu_subgraph import BuildAipuSubgraph
from .extract_in_out_quant_ops import ExtractInOutQuantOps
from .quantization_helper_pre import QuantizationHelperPre
from .quantization_helper_post import QuantizationHelperPost
from .set_node_compiler_to_default import SetNodeCompilerToDefault
from .extract_cellstate_hiddenstate_to_tuple import ExtractCellStateAndHiddenStateToTupleOutput

try:
    from .fuse_ops import AIPUFuseOp
except ImportError:
    import tvm

    @tvm.ir.transform.module_pass(opt_level=0)
    class AIPUFuseOp:
        def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
            return mod


try:
    from .compile_fused_op import CompileFusedOp, generate_op_lib_impl
except ImportError:
    from tvm import relay

    @relay.transform.function_pass(opt_level=0)
    class CompileFusedOp:
        def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
            return func

    def generate_op_lib_impl(dst_dir, target="X2_1204"):  # pylint: disable=unused-argument
        return
