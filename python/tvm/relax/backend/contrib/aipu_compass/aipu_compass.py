# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Simple wrap of AIPU Compass compile flow."""
import os
import time
import tvm
from tvm import relax
from tvm.script import from_source
from tvm.relax.op.contrib.aipu_compass.pattern_table import pattern_table_pre
from tvm.aipu.logger import timer, set_logger, DEBUG_ONCE, INFO
from .config import config_aipu_compass, AipuCompassConfig
from .parser import parse_model
from . import transform as compass_transform
from . import analysis as compass_analysis
from .deployable import Deployable


class AipuCompass:
    """The class which organizes all compilation relevant API functions.

    Attributes
    ----------
    ir_mod : tvm.relax.irmod
        The Relax IR between each compilation phase.
    """

    def __init__(self, config):
        """The constructor of this class.

        The configuration file will be parsed and processed,
        and then stores all of the configurations to the
        global singleton of Python class AipuCompassConfig.

        Parameters
        ----------
        config: str
            The path of the configuration file.
        """
        self.ir_mod = None

        config_aipu_compass(config)
        cfg = AipuCompassConfig.get().common
        self._disabled_pass = []
        for pass_name in cfg["disabled_pass"].strip("[]").split(","):
            pass_name = pass_name.strip()
            if pass_name != "" and pass_name not in self._disabled_pass:
                self._disabled_pass.append(pass_name)

        set_logger(cfg["log_file"], cfg["log_level"])
        DEBUG_ONCE(f"TVM {tvm.__version__} ({os.path.dirname(tvm.__file__)})")
        INFO(f"Current Output Path: {cfg['output_dir']}")

    @timer
    def parse(self, update_params=None):  # pylint: disable=unused-argument
        """The parse phase of the compilation flow.

        Parameters
        ----------
        update_params: dict
            The extra parameters need to be updated to the
            parameters of the NN model, such as binding an
            input of the NN model to a specific constant value.
        """
        parser_config = AipuCompassConfig.get().parser
        ir_mod = parse_model(parser_config)
        # # Use parameters (constant node) to replace the corresponding variable
        # # node, before this all of inputs and weights of neural network graph
        # # are parameters of function "main", after this only inputs of neural
        # # network graph still are parameters of function "main".
        # # After this statement, the "params" isn't needed, because it is merged
        # # into "ir_mod".
        # if update_params:
        #     params.update(update_params)
        # ir_mod["main"] = relay.build_module.bind_params_by_name(ir_mod["main"], params)
        self.ir_mod = ir_mod

    @timer
    def optimize(self):
        """The optimize phase of the compilation flow."""
        cfg = AipuCompassConfig.get().common
        is_quant_model = compass_analysis.has_quantized_op(self.ir_mod)
        if is_quant_model:
            cfg["compat_quantized_model"] = "true"

        # Simplify and general optimizations.
        passes = [
            relax.transform.DecomposeOpsForInference(),
            relax.transform.FoldConstant(),
            compass_transform.HintPatternRewrite(),
            relax.transform.RemoveUnusedOutputs(),
        ]

        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

    @timer
    def _specific_optimize(self):
        """AIPU Compass specific Optimization."""
        desired_layouts = {
            "relax.nn.conv2d": ["NHWC", "OHWI"],
            "relax.nn.conv2d_transpose": ["NHWC", "OHWI"],
        }
        passes = [
            relax.transform.ConvertLayout(desired_layouts),
            # Need before FoldConstant and after ConvertLayout.
            compass_transform.SinkDequantize(),
            relax.transform.FoldConstant(),
            compass_transform.SinkTranspose(),
            compass_transform.ConvertOps(),
            compass_transform.HintPatternRewrite(),
            compass_transform.RevertTranspose(),
        ]

        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

    @timer
    def _partition(
        self, postprocess_hint_fn=None, fallback_indices=None
    ):  # pylint: disable=unused-argument
        passes = [
            # Need before FuseOpsByPattern. Don't use pattern rewrite after this pass.
            compass_transform.CopyDequantize(),
            relax.transform.FuseOpsByPattern(pattern_table_pre()),
            compass_transform.FuseTuple(),
            relax.transform.MergeCompositeFunctions(),
            compass_transform.RenameAipuSubfunc(),
            compass_transform.SaveInpQuantInfo(),
        ]
        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

        # todo:
        #     dump_annotation_graph
        #     SetNodeCompilerToDefault
        #     RearrangeParams
        #     dump_partitioning_graph
        #     PruneAIPUSubGraphs
        #     RearrangeNames

    # todo
    # @timer
    # def _restore_layout(self):
    #     """Restore the layout of the operators left in Relay main function."""

    @timer
    def partition(self, postprocess_hint_fn=None, fallback_indices=None):
        """The partition phase of the compilation flow.

        Partition the graph greedily for offloading supported operators to the Zhouyi NPU.

        Parameters
        ----------
        postprocess_hint_fn : Function of F(expr) -> bool
            A hint function to help find postprocess of model

        fallback_indices : list of int
            The indices of nodes in Relay IR if want to fallback to cpu.

        """
        self._specific_optimize()
        self._partition(postprocess_hint_fn=postprocess_hint_fn, fallback_indices=fallback_indices)
        # self._restore_layout()

        # Always dump the snapshoot of the final partitioned graph.
        output_dir = AipuCompassConfig.get().common["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        self.save(f"{output_dir}/partitioned_graph.txt", False)

    def build_aipu_subgraphs(self, bypass_input_quant=None, bypass_output_dequant=None):
        """Build all AIPU subgraphs before Relay build the whole model.

        This API will store the build results in Relax IR at this time, these results will be used
        to create runtime module for their corresponding AIPU subgraphs when Relay build the whole
        model. It will be invoked by the "build" API automatically, but it also can be used alone.

        Parameters
        ----------
        bypass_input_quant : List[int]
          List of index, indicate which inputs bypass quantize op insertion.

        bypass_output_dequant : List[int]
          List of index, indicate which outputs bypass dequantize op insertion.

        Returns
        -------
        """

        if any(hasattr(x.attrs, "compass.pre_build") for x in self.ir_mod.functions.values()):
            return

        t_start = time.perf_counter()
        INFO("build_aipu_subgraphs start")

        cfg = AipuCompassConfig.get().common

        if bypass_input_quant is None:
            bypass_input_quant = []
        if bypass_output_dequant is None:
            bypass_output_dequant = []

        passes = [
            # todo compass_transform.ExtractInOutQuantOps(),
            compass_transform.BuildAipuSubgraph(
                cfg["forward_engine"], bypass_input_quant, bypass_output_dequant
            ),
        ]
        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

        # if cfg["dump_building_graph"] == "true":
        #     output_dir = cfg["output_dir"]
        #     os.makedirs(output_dir, exist_ok=True)
        #     self.save(f"{output_dir}/after_build_aipu_subgraphs.txt", False)

        INFO(f"build_aipu_subgraphs finished, elapsed time: {(time.perf_counter() - t_start):.2f}s")

    @timer
    def _build(self, target="llvm"):
        with tvm.transform.PassContext(3, disabled_pass=self._disabled_pass):
            self.ir_mod = relax.transform.RunCodegen()(self.ir_mod)
            compiled_model = relax.build(self.ir_mod, target)
            return Deployable(compiled_model)

    @timer
    def build(
        self,
        target="llvm",
        bypass_input_quant=None,
        bypass_output_dequant=None,
    ):
        """The build phase of the compilation flow.

        Parameters
        ----------
        target : str, or tvm.target.Target,
                 or dict of str(i.e. device name) to str/tvm.target.Target
            For heterogeneous compilation, it is a dictionary indicating
            the context to target mapping.
            For homogeneous compilation, it is a build target.
            It will be passed to relax.build.

        bypass_input_quant : List[int]
            List of index, indicate which inputs bypass quantize op insertion.

        bypass_output_dequant : List[int]
            List of index, indicate which outputs bypass dequantize op insertion.

        Returns
        -------
        result : deployable.Deployable
            The compiled NN model that can be exported and deployed.
        """
        self.build_aipu_subgraphs(bypass_input_quant, bypass_output_dequant)
        return self._build(target=target)

    def compile(
        self,
        update_params=None,
        target="llvm",
        postprocess_hint_fn=None,
        fallback_indices=None,
        bypass_input_quant=None,
        bypass_output_dequant=None,
    ):
        """The all-in-one API of the compilation flow.

        Parameters
        ----------
        update_params : dict
            The extra parameters need to be updated to the parameters of the NN model,
            such as binding an input of the NN model to a specific constant value.

        target : str, or tvm.target.Target,
                 or dict of str(i.e. device name) to str/tvm.target.Target
            For heterogeneous compilation, it is a dictionary indicating
            the context to target mapping.
            For homogeneous compilation, it is a build target.
            It will be passed to relax.build.

        postprocess_hint_fn : Function of F(expr) -> bool
            A hint function to help find postprocess of the model.

        fallback_indices : list of int
            The indices of nodes in Relax IR if want to fallback to cpu.

        bypass_input_quant : List[int]
            List of index, indicate which inputs bypass quantize op insertion.

        bypass_output_dequant : List[int]
            List of index, indicate which outputs bypass dequantize op insertion.

        Returns
        -------
        result : deployable.Deployable
            The compiled NN model that can be exported and deployed.
        """
        # 1. Parse the nn model to Relax IR.
        self.parse(update_params)
        # 2. Optimize the nn model for AIPU Compass.
        self.optimize()
        # 3. Partition the nn model for AIPU Compass.
        self.partition(postprocess_hint_fn, fallback_indices)
        # todo 4. Collect calibration data for all AIPU Compass functions if needed.
        # self.collect_calibration_data()
        # 5. Build the nn model.
        return self.build(
            target=target,
            bypass_input_quant=bypass_input_quant,
            bypass_output_dequant=bypass_output_dequant,
        )

    def save(self, path, show_meta=True):
        with open(path, "w") as f:
            f.write(self.ir_mod.script(show_meta=show_meta))

    def load(self, path):
        with open(path) as f:
            self.ir_mod = from_source(f.read())
