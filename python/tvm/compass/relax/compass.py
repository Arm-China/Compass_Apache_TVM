# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Simple wrap of Zhouyi Compass compile flow."""
import os
import time
from torch.utils.data import DataLoader as PyTorchDataLoader
import tvm
from tvm import relax, script
from ..logger import timer, set_logger, DEBUG_ONCE, INFO
from . import analysis as compass_analysis
from . import transform as compass_transform
from .op.pattern_table import FLOAT_PATTERNS, QUANT_PATTERNS
from .config import config_compass, CompassConfig
from .parser import parse_model
from .engine import create_calibrate_collector
from .deployable import Deployable
from .builder import create_dataset
from .utils import X86_DESIRED_LAYOUTS


class Compass:
    """The class which organizes all compilation relevant API functions.

    Attributes
    ----------
    ir_mod : tvm.relax.irmod
        The Relax IR between each compilation phase.
    """

    def __init__(self, config):
        """The constructor of this class.

        The configuration file will be parsed and processed, and then stores all of the
        configurations to the global singleton of Python class ``CompassConfig``.

        Parameters
        ----------
        config: str
            The path of the configuration file.
        """
        self.ir_mod = None

        config_compass(config)
        cfg = CompassConfig.get().common
        self._disabled_pass = []
        for pass_name in cfg["disabled_pass"].strip("[]").split(","):
            pass_name = pass_name.strip()
            if pass_name != "" and pass_name not in self._disabled_pass:
                self._disabled_pass.append(pass_name)

        set_logger(cfg["log_file"], cfg["log_level"])
        DEBUG_ONCE(f"TVM {tvm.__version__} ({os.path.dirname(tvm.__file__)})")
        INFO(f"Current Output Path: {cfg['output_dir']}")

    @timer
    def parse(self, update_params=None):
        """The parse phase of the compilation flow.

        Parameters
        ----------
        update_params: dict
            The extra parameters need to be updated to the
            parameters of the NN model, such as binding an
            input of the NN model to a specific constant value.
        """
        parser_config = CompassConfig.get().parser
        ir_mod = parse_model(parser_config)
        if update_params:
            # Use constant data to replace the corresponding parameters of function "main".
            ir_mod = relax.transform.BindParams("main", update_params)(ir_mod)
        self.ir_mod = ir_mod

    @timer
    def optimize(self):
        """The optimize phase of the compilation flow."""
        cfg = CompassConfig.get().common
        is_quant_model = compass_analysis.has_quantized_op(self.ir_mod)
        if is_quant_model:
            cfg["compat_quantized_model"] = "true"

        # Simplify and general optimizations.
        passes = [
            relax.transform.RemoveUnusedOutputs(),
            compass_transform.LegalizeVarName(),
            relax.transform.EliminateCommonSubexpr(),
            relax.transform.DecomposeOpsForInference(),
            relax.transform.FoldConstant(),
            compass_transform.PatternRewriteInOptimize(),
            compass_transform.RewriteOpsInOptimize(),
            relax.transform.RemoveUnusedOutputs(),
        ]

        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

    @timer
    def _specific_optimize(self):
        """Compass specific Optimization."""
        desired_layouts = {
            "relax.nn.conv2d": ["NHWC", "OHWI"],
            "relax.nn.conv2d_transpose": ["NHWC", "OHWI"],
            "relax.image.resize2d": ["NHWC"],
            "relax.nn.max_pool2d": ["NHWC"],
            "relax.nn.avg_pool2d": ["NHWC"],
            "relax.nn.space_to_depth": ["NHWC"],
            "relax.nn.depth_to_space": ["NHWC"],
            "relax.channel_shuffle": ["NHWC"],
        }
        passes = [
            compass_transform.PatternRewriteBeforeConvertLayout(),
            relax.transform.ConvertLayout(desired_layouts),
            # Need before FoldConstant and after ConvertLayout.
            compass_transform.SinkDequantize(),
            relax.transform.FoldConstant(),
            compass_transform.SinkTranspose(),
            compass_transform.PatternRewriteAfterConvertLayout(),
            compass_transform.RevertTranspose(),
        ]

        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

    @timer
    def _partition(
        self, postprocess_hint_fn=None, fallback_nodes=None
    ):  # pylint: disable=unused-argument
        cfg = CompassConfig.get().common
        is_quant = cfg["compat_quantized_model"] == "true"
        pattern_table = QUANT_PATTERNS + FLOAT_PATTERNS if is_quant else FLOAT_PATTERNS

        passes = [
            compass_transform.GetPostProcessFunction(postprocess_hint_fn),
            # Need before FuseOpsByPattern. Don't use pattern rewrite after this pass.
            compass_transform.CopyDequantize(),
            relax.transform.FuseOpsByPattern(pattern_table),
            compass_transform.PatternRewriteAfterFuseOps(),
            compass_transform.FuseTuple(),
        ]
        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

        if cfg["dump_annotation_graph"] == "true":
            # pylint: disable=not-callable
            self.ir_mod = compass_transform.UniqueVarName()(self.ir_mod)
            output_dir = cfg["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            # Dump the graph snapshoot before merge composite functions stage.
            self.save(f"{output_dir}/partitioning_annotation_graph.txt", False)

        passes = [
            compass_transform.FallbackNodesToCPU(fallback_nodes),
            relax.transform.MergeCompositeFunctions(),
            compass_transform.RenameCompassSubfunc(),
            compass_transform.InlineSingleOpFunc(),
            compass_transform.ConvertCompassOps(),
            compass_transform.SaveInpQuantInfo(),
            compass_transform.RearrangeParams(),
        ]
        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

        if cfg["dump_partitioning_graph"] == "true":
            output_dir = cfg["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            # Dump the graph snapshoot of the 1st partitioning stage.
            self.save(f"{output_dir}/partitioning_graph0.txt", False)

        passes = [
            compass_transform.PruneCompassSubGraphs(),
        ]
        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

    @timer
    def _restore_layout(self):
        """Restore the layout of the operators left in Relax main function."""
        passes = [
            compass_transform.PatternRewriteAfterPartition(),
            compass_transform.ConvertLayout("main", X86_DESIRED_LAYOUTS),
            compass_transform.SinkTranspose(),
            relax.transform.FoldConstant(),
        ]
        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

    @timer
    def partition(self, postprocess_hint_fn=None, fallback_nodes=None):
        """The partition phase of the compilation flow.

        Partition the graph greedily for offloading supported operators to the Zhouyi NPU.

        Parameters
        ----------
        postprocess_hint_fn : Function of F(expr) -> bool
            A hint function to help find postprocess of model

        fallback_nodes : dict
            Fallback nodes to cpu if needed.
            key: str
                the name of caller function.
            value: list of str
                The name of binding vars in caller function.

        """
        self._specific_optimize()
        self._partition(postprocess_hint_fn=postprocess_hint_fn, fallback_nodes=fallback_nodes)
        self._restore_layout()

        # Always dump the snapshoot of the final partitioned graph.
        output_dir = CompassConfig.get().common["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        self.save(f"{output_dir}/partitioned_graph.txt", False)

    def collect_calibration_data(self):
        """The collect calibration data phase of the compilation flow.

        This API will collect the calibration data used by each NPU sub graph
        by executing the NN model with the user supplied calibration data.
        The generated dataset will be saved to the path xxx/yyy/optimizer/calibration_data.npz
        (xxx is the root output directory and yyy is the NPU subgraph name).
        It will be used automatically by the Zhouyi Compass NN compiler during the build phase.
        """
        common_cfg = CompassConfig.get().common
        # Skip when there only is 1 Compass function and each argument of Compass function is same
        # with the corresponding parameter of main function, i.e., the Compass function is the
        # bellwether of the whole IRModule.
        if (
            compass_analysis.check_bellwether(self.ir_mod, "compass")
            or common_cfg["compat_quantized_model"] == "true"
        ):
            return

        t_start = time.perf_counter()
        INFO("collect_calibration_data start")

        # Compile the temporary model.
        calibrate_collector = create_calibrate_collector(common_cfg["calibrate_collector"])
        with calibrate_collector:
            ir_mod = relax.transform.RunCodegen()(self.ir_mod)
            run_wrapper = relax.VirtualMachine(relax.build(ir_mod, "llvm"), tvm.cpu())["main"]

        # Run the compiled model using user provided calibration dataset.
        input_shapes = []
        for param in self.ir_mod["main"].params:
            ttype = param.struct_info
            if isinstance(ttype, relax.TupleStructInfo):
                for t in ttype.fields:
                    input_shapes.append([int(d) for d in t.shape])
            else:
                input_shapes.append([int(d) for d in ttype.shape])
        batch_size = input_shapes[0][0]
        opt_cfg = CompassConfig.get().optimizer
        calibrate_dataset = create_dataset(opt_cfg["dataset"], opt_cfg["calibration_data"])
        if common_cfg["without_batch_dim"] == "true":
            minibatch_size = 1
        else:
            minibatch_size = batch_size
        data_loader = PyTorchDataLoader(calibrate_dataset, batch_size=minibatch_size)
        for minibatch_features, _ in data_loader:
            if not isinstance(minibatch_features, list):
                minibatch_features = [minibatch_features]
            args = []
            for i, arg in enumerate(minibatch_features):
                arg = arg.numpy()
                input_shape = input_shapes[i]
                if arg.shape == tuple(input_shape):
                    args.append(arg)
                elif arg.squeeze().shape == tuple(x for x in input_shape if x != 1):
                    args.append(arg.reshape(input_shape))
                else:
                    raise ValueError(
                        "Get dataset failed. Please make sure your dataset has batch dim"
                        " and check model input shape."
                    )
                # Ensure the data type of arguments match that of parameters.
                param_dtype = self.ir_mod["main"].params[i].struct_info.dtype
                arg_dtype = str(arg.dtype)
                if param_dtype != arg_dtype:
                    raise TypeError(
                        f"The type of the {i}th input in dataset mismatches that of the model, "
                        f'expected: "{param_dtype}" vs real: "{arg_dtype}".'
                    )
            run_wrapper(*args)

        if isinstance(calibrate_dataset.data, list):
            dataset_batch_size = calibrate_dataset.data[0].shape[0]
        else:
            dataset_batch_size = calibrate_dataset.data.shape[0]

        calibrate_collector.finish(dataset_batch_size)
        INFO(
            f"collect_calibration_data finished, "
            f"elapsed time: {(time.perf_counter() - t_start):.2f}s"
        )

    def build_compass_subgraphs(self, target="llvm", bypass_output_dequant=None):
        """Build all Compass subgraphs before Relax build the whole model.

        This API will store the build results in Relax IR at this time, these results will be used
        to create runtime module for their corresponding Compass subgraphs when Relax build the
        whole model. It will be invoked by the "build" API automatically, but it also can be used
        alone.

        Parameters
        ----------
        target : str, or tvm.target.Target
                 or dict of str(i.e. device name) to str/tvm.target.Target
            For heterogeneous compilation, it is a dictionary indicating
            the context to target mapping.
            For homogeneous compilation, it is a build target.
            It will be passed to relax.build.

        bypass_output_dequant : Optional[List[int]]
            List of index, indicate which outputs bypass dequantize op insertion.

        Returns
        -------
        """

        if any(hasattr(x.attrs, "compass.pre_build") for x in self.ir_mod.functions.values()):
            return

        t_start = time.perf_counter()
        INFO("build_compass_subgraphs start")

        cfg = CompassConfig.get().common

        if bypass_output_dequant is None:
            bypass_output_dequant = []

        passes = []
        if CompassConfig.get().gbuilder["target"].startswith(("Z", "X1")):
            passes = [
                compass_transform.ExtractInOutQuantOps(),
                compass_transform.SaveInpQuantInfo(),
            ]
        passes += [
            compass_transform.BuildCompassSubgraph(
                cfg["forward_engine"], target, bypass_output_dequant
            ),
        ]
        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

        if cfg["dump_building_graph"] == "true":
            output_dir = cfg["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            self.save(f"{output_dir}/after_build_compass_subgraphs.txt", False)

        elapsed_time = time.perf_counter() - t_start
        INFO(f"build_compass_subgraphs finished, elapsed time: {elapsed_time:.2f}s")

    @timer
    def _build(self, target="llvm"):
        with tvm.transform.PassContext(3, disabled_pass=self._disabled_pass):
            self.ir_mod = relax.transform.RunCodegen()(self.ir_mod)
            compiled_model = relax.build(self.ir_mod, target)
            return Deployable(compiled_model.mod)

    @timer
    def build(self, target="llvm", bypass_output_dequant=None):
        """The build phase of the compilation flow.

        Parameters
        ----------
        target : str, or tvm.target.Target,
                 or dict of str(i.e. device name) to str/tvm.target.Target
            For heterogeneous compilation, it is a dictionary indicating
            the context to target mapping.
            For homogeneous compilation, it is a build target.
            It will be passed to relax.build.

        bypass_output_dequant : Optional[List[int]]
            List of index, indicate which outputs bypass dequantize op insertion.

        Returns
        -------
        result : deployable.Deployable
            The compiled NN model that can be exported and deployed.
        """
        self.build_compass_subgraphs(target, bypass_output_dequant)
        return self._build(target=target)

    def compile(
        self,
        update_params=None,
        target="llvm",
        postprocess_hint_fn=None,
        fallback_nodes=None,
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

        fallback_nodes : dict
            Fallback nodes to cpu if needed.
            key: str
                the name of caller function.
            value: list of str
                The name of binding vars in caller function.

        bypass_output_dequant : Optional[List[int]]
            List of index, indicate which outputs bypass dequantize op insertion.

        Returns
        -------
        result : deployable.Deployable
            The compiled NN model that can be exported and deployed.
        """
        # 1. Parse the NN model to Relax IR.
        self.parse(update_params)
        # 2. Optimize the NN model for Compass.
        self.optimize()
        # 3. Partition the NN model for Compass.
        self.partition(postprocess_hint_fn, fallback_nodes)
        # 4. Collect calibration data for all Compass functions if needed.
        self.collect_calibration_data()
        # 5. Build the NN model.
        return self.build(target=target, bypass_output_dequant=bypass_output_dequant)

    def save(self, path, show_meta=True):
        with open(path, "w") as f:
            f.write(self.ir_mod.script(show_meta=show_meta))

    def load(self, path):
        with open(path) as f:
            self.ir_mod = script.from_source(f.read())
