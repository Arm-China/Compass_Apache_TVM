# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
"""Simple wrap of AIPU Compass compile flow."""
import os
import time
from torch.utils.data import DataLoader as PyTorchDataLoader
import tvm
from tvm import relay
from tvm.relay.backend import Runtime, Executor
from tvm.relay.op.contrib.aipu_compass.pattern_table import pattern_table_pre, pattern_table_post
from tvm.aipu.logger import timer, set_logger, INFO, WARN
from .config import config_aipu_compass, AipuCompassConfig, AipuCompassBasicConfig
from .parser import parse_model
from . import transform as compass_transform
from . import analysis as compass_analysis
from .engine import create_calibrate_collector
from .deployable import Deployable
from .aipu_builder import create_dataset
from . import utils
from . import _ffi_api


def _get_executor_name():
    # Select the executor which determine how to build and run the Relay model.
    value = AipuCompassConfig.get().common["executor"]
    if value in ("vm", "graph"):
        return value
    raise ValueError(f'Invalid AIPU Compass executor "{value}".')


class AipuCompass:
    """The class which organizes all compilation relevant API functions.

    Attributes
    ----------
    ir_mod : tvm.relay.irmod
        The Relay IR between each compilation phase.
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
        parser_config = AipuCompassConfig.get().parser
        ir_mod, params = parse_model(parser_config)
        # Use parameters (constant node) to replace the corresponding variable
        # node, before this all of inputs and weights of neural network graph
        # are parameters of function "main", after this only inputs of neural
        # network graph still are parameters of function "main".
        # After this statement, the "params" isn't needed, because it is merged
        # into "ir_mod".
        if update_params:
            params.update(update_params)
        ir_mod["main"] = relay.build_module.bind_params_by_name(ir_mod["main"], params)
        self.ir_mod = ir_mod

    @timer
    def optimize(self):
        """The optimize phase of the compilation flow."""
        # Simplify and general optimizations.
        passes = [
            compass_transform.LegalizeVarName(),
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.EliminateCommonSubexpr(),
            relay.transform.SimplifyInference(),
            compass_transform.QuantizationHelperPre(),
            relay.transform.FakeQuantizationToInteger(),
            compass_transform.QuantizationHelperPost(),
            relay.transform.FoldConstant(),
            compass_transform.LoopStructureMerger(),
            relay.transform.RemoveUnusedFunctions(),
            compass_transform.VerifyLetLoopParams(),
            compass_transform.UnrollLetLoopInMain(),
            relay.transform.FoldConstant(),
            compass_transform.EliminateTensorArrayOp(),
            compass_transform.EvaluateZeroFreeArgsCall(),
            relay.transform.FoldConstant(),
            relay.transform.DynamicToStatic(),
            relay.transform.EliminateCommonSubexpr(),
            relay.transform.SimplifyExpr(),
            relay.transform.FoldConstant(),
            relay.transform.FoldScaleAxis(),
            relay.transform.CanonicalizeOps(),
            relay.transform.FoldExplicitPadding(),
            compass_transform.EvaluateZeroFreeArgsCall(),
            relay.transform.FoldConstant(),
        ]

        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

    @timer
    def _specific_optimize(self):
        """AIPU Compass specific Optimization."""
        desired_layouts = {
            "nn.conv2d": ["NHWC", "OHWI"],
            "qnn.conv2d": ["NHWC", "OHWI"],
            "nn.conv3d": ["NDHWC", "OHWDI"],
            "nn.conv2d_transpose": ["NHWC", "OHWI"],
            "qnn.conv2d_transpose": ["NHWC", "OHWI"],
            "nn.adaptive_avg_pool1d": ["NWC"],
            "nn.max_pool2d": ["NHWC"],
            "nn.avg_pool2d": ["NHWC"],
            "nn.global_avg_pool2d": ["NHWC"],
            "nn.global_max_pool2d": ["NHWC"],
            "nn.upsampling": ["NHWC"],
            "image.resize2d": ["NHWC"],
            "image.grid_sample": ["NHWC"],
            "image.crop_and_resize": ["NHWC"],
            "reverse": ["NHWC"],
            "nn.space_to_depth": ["NHWC"],
            "nn.depth_to_space": ["NHWC"],
            "vision.roi_align": ["NHWC", "default"],
            "vision.roi_pool": ["NHWC", "default"],
            "nn.deformable_conv2d": ["NHWC", "HWIO"],
            "contrib.aipu_compass.deformable_conv2d_v2": ["NHWC", "HWIO"],
            "nn.adaptive_avg_pool2d": ["NHWC"],
            "contrib.aipu_compass.channel_shuffle": ["NHWC"],
        }
        passes = [
            compass_transform.HintPatternRewrite(),
            relay.transform.FoldScaleAxis(),
            relay.transform.ConvertLayout(desired_layouts),
            compass_transform.HintPatternRewrite(),
            relay.transform.FoldConstant(),
            compass_transform.PrimalLayoutTransformToTranspose(),
            compass_transform.SinkTranspose(),
            compass_transform.ExtractNegativePadFromConv2d(),
            relay.transform.SimplifyExpr(),
            relay.transform.FoldConstant(),
            compass_transform.SplitDeformableConv2d(),
            compass_transform.ExtractCellStateAndHiddenStateToTupleOutput(),
        ]

        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

    @timer
    def _partition(self, postprocess_hint_fn=None, fallback_indices=None):
        cfg = AipuCompassConfig.get().common

        include_float, include_quant = True, False
        is_quant_model = compass_analysis.has_quantized_op(self.ir_mod)
        if is_quant_model:
            include_float, include_quant = False, True
            cfg["compat_quantized_model"] = "true"
            _ffi_api.OpRegistry_SwapAttrMap("target.aipu_compass", "target.aipu_compass.qnn")

        passes = [
            relay.transform.MergeComposite(pattern_table_pre(include_float, include_quant)),
            compass_transform.ConvertAIPUOps(),
            relay.transform.MergeComposite(pattern_table_post(include_float, include_quant)),
            relay.transform.SimplifyExpr(),
            relay.transform.FoldConstant(),
            compass_transform.GetPostProcessFunction(postprocess_hint_fn),
            compass_transform.AIPUFuseOp(),
            compass_transform.CompileFusedOp(),
            relay.transform.AnnotateTarget("aipu_compass"),
            compass_transform.ReAnnotateTuple(),
        ]
        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

        if cfg["dump_annotation_graph"] == "true":
            output_dir = cfg["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            # Dump the graph snapshoot after annotation stage.
            self.save(f"{output_dir}/partitioning_annotation_graph.txt", False)

        passes = [
            compass_transform.SetNodeCompilerToDefault(fallback_indices),
            relay.transform.MergeCompilerRegions(),
            relay.transform.PartitionGraph(),
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
            compass_transform.PruneAIPUSubGraphs(),
            compass_transform.RearrangeNames(),
        ]
        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

        if is_quant_model:
            _ffi_api.OpRegistry_SwapAttrMap("target.aipu_compass", "target.aipu_compass.qnn")

    @timer
    def _restore_layout(self):
        """Restore the layout of the operators left in Relay main function."""
        passes = [
            relay.transform.ConvertLayout(utils.X86_DESIRED_LAYOUTS),
            relay.transform.FoldConstant(),
            compass_transform.PrimalLayoutTransformToTranspose(),
            compass_transform.SinkTranspose(),
            relay.transform.SimplifyExpr(),
            relay.transform.FoldConstant(),
        ]
        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

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
        self._restore_layout()

        # Always dump the snapshoot of the final partitioned graph.
        output_dir = AipuCompassConfig.get().common["output_dir"]
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
        common_cfg = AipuCompassConfig.get().common
        # Skip when there only is 1 AIPU Compass function and each argument of
        # AIPU Compass function is same with the corresponding parameter of main
        # function, i.e., the AIPU Compass function is the bellwether of the
        # whole IRModule.
        if (
            compass_analysis.check_bellwether(self.ir_mod, "aipu_compass")
            or common_cfg["compat_quantized_model"] == "true"
        ):
            return

        t_start = time.perf_counter()
        INFO("collect_calibration_data start")

        # Compile the temporary model.
        calibrate_collector = create_calibrate_collector(common_cfg["calibrate_collector"])
        with calibrate_collector:
            run_wrapper = relay.create_executor(kind="vm", mod=self.ir_mod).evaluate()

        # Run the compiled model using user provided calibration dataset.
        input_shapes = []
        for param in self.ir_mod["main"].params:
            ttype = param.checked_type
            if isinstance(ttype, relay.TupleType):
                for t in ttype.fields:
                    input_shapes.append([int(d) for d in t.shape])
            else:
                input_shapes.append([int(d) for d in ttype.shape])
        batch_size = input_shapes[0][0]
        opt_cfg = AipuCompassConfig.get().optimizer
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
                param_dtype = self.ir_mod["main"].params[i].checked_type.dtype
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

    def build_aipu_subgraphs(self, bypass_input_quant=None, bypass_output_dequant=None):
        """Build all AIPU subgraphs before Relay build the whole model.

        This API will store the build results in Relay IR at this time, these results will be used
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
            compass_transform.ExtractInOutQuantOps(),
            compass_transform.BuildAipuSubgraph(
                cfg["forward_engine"], bypass_input_quant, bypass_output_dequant
            ),
        ]
        with tvm.transform.PassContext(opt_level=3, disabled_pass=self._disabled_pass):
            self.ir_mod = tvm.transform.Sequential(passes)(self.ir_mod)

        if cfg["dump_building_graph"] == "true":
            output_dir = cfg["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            self.save(f"{output_dir}/after_build_aipu_subgraphs.txt", False)

        INFO(f"build_aipu_subgraphs finished, elapsed time: {(time.perf_counter() - t_start):.2f}s")

    @timer
    def _build(self, target="llvm", executor=None, runtime=None):
        with tvm.transform.PassContext(3, disabled_pass=self._disabled_pass):
            if _get_executor_name() == "vm":
                if executor or runtime:
                    WARN('Parameter "executor" and "runtime" will be ignored when build with VM.')
                compiled_model = relay.vm.compile(self.ir_mod, target=target)
            else:
                executor = executor or Executor("graph")
                runtime = runtime or Runtime("cpp")
                compiled_model = relay.build(
                    self.ir_mod, target=target, executor=executor, runtime=runtime
                )

            return Deployable(compiled_model)

    @timer
    def build(
        self,
        target="llvm",
        executor=None,
        runtime=None,
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
            It will be passed to relay.build or relay.vm.compile.

        executor : tvm.relay.backend.Executor
            The executor configuration with which to build the model.
            Defaults to "graph" if no executor specified.

        runtime : tvm.relay.backend.Runtime
            Runtime configuration to use when building the model.
            Defaults to "cpp" if no runtime specified.

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
        return self._build(target=target, executor=executor, runtime=runtime)

    def compile(
        self,
        update_params=None,
        target="llvm",
        postprocess_hint_fn=None,
        fallback_indices=None,
        executor=None,
        runtime=None,
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
            It will be passed to relay.build or relay.vm.compile.

        postprocess_hint_fn : Function of F(expr) -> bool
            A hint function to help find postprocess of the model.

        fallback_indices : list of int
            The indices of nodes in Relay IR if want to fallback to cpu.

        executor : tvm.relay.backend.Executor
            The executor configuration with which to build the model.
            Defaults to "graph" if no executor specified.

        runtime : tvm.relay.backend.Runtime
            Runtime configuration to use when building the model.
            Defaults to "cpp" if no runtime specified.

        bypass_input_quant : List[int]
            List of index, indicate which inputs bypass quantize op insertion.

        bypass_output_dequant : List[int]
            List of index, indicate which outputs bypass dequantize op insertion.

        Returns
        -------
        result : deployable.Deployable
            The compiled NN model that can be exported and deployed.
        """
        # 1. Parse the nn model to Relay IR.
        self.parse(update_params)
        # 2. Optimize the nn model for AIPU Compass.
        self.optimize()
        # 3. Partition the nn model for AIPU Compass.
        self.partition(postprocess_hint_fn, fallback_indices)
        # 4. Collect calibration data for all AIPU Compass functions if needed.
        self.collect_calibration_data()
        # 5. Build the nn model.
        return self.build(
            target=target,
            executor=executor,
            runtime=runtime,
            bypass_input_quant=bypass_input_quant,
            bypass_output_dequant=bypass_output_dequant,
        )

    def save(self, path, show_meta_data=True):
        with open(path, "w") as f:
            f.write(self.ir_mod.astext(show_meta_data))

    def load(self, path):
        with open(path) as f:
            self.ir_mod = relay.fromtext(f.read())


def sync_compass_output_dir(rpc_sess):
    """Synchronize files of compass output directory on RPC server to local.

    Parameters
    ----------
    rpc_sess : tvm.rpc.RPCSession
        The RPC session that is already connected to the RPC server.
    """
    if rpc_sess is None:
        return
    remote_files = [x for x in rpc_sess.list_files(".") if x.startswith("compass_output")]

    local_output_dir = AipuCompassBasicConfig.get().common["output_dir"]
    for remote_file in remote_files:
        rel_path = remote_file.split(os.path.sep, 1)[1]
        with open(os.path.join(local_output_dir, rel_path), "wb") as f:
            f.write(rpc_sess.download(remote_file))
            INFO(f'Downloaded "{rel_path}" into "{local_output_dir}".')
