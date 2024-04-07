# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Split module for pipelines."""
import copy
import os
from threading import Thread
import json
import textwrap
import tvm
from tvm import relay
from tvm._ffi import get_global_func
from tvm.contrib import pipeline_executor_build, graph_executor
from tvm.contrib.pipeline_executor import PipelineExecutorFactoryModule
from torch.utils.data import DataLoader as PyTorchDataLoader
import numpy as np
from .config import AipuCompassConfig
from .aipu_compass import AipuCompass
from .aipu_builder import create_dataset


class CompassPipelineExecutorFactoryModule(PipelineExecutorFactoryModule):
    """
    Wrapper for PipelineExecutorFactoryModule class
    override export_library function to export shared_config
    """

    def __init__(self, pipeline_mods, mods_config, shared_config):
        """Init the pipeline factory, add shared_config"""

        super(CompassPipelineExecutorFactoryModule, self).__init__(pipeline_mods, mods_config)
        self.shared_config = shared_config

    def compass_graph_executor_create(self, pipeline_mods, mod_config, shared_config):
        """Create graph_executor list and return configuration as a json string.

        Parameters
        ----------
        pipeline_mods : List[GraphExecutorFactoryModule]
          List of GraphExecutorFactoryModule

        mod_config : Dict[str, Any]
            Modules dependency configuration information.

        shared_config : Dict[str, List[str]]
            Compass Module shared tensor information.

        Returns
        -------
        mods : List[Module]
            The Module list.

        mod_config : str
            The Modudle configuration.
        """
        # Should store modules in the list named 'mods' in index order.
        mods = [None for _ in range(len(pipeline_mods))]

        factory_mods = [pipeline_mods[idx]["lib"].module for idx in pipeline_mods]
        reconfig_fn = get_global_func(
            "compass_pipeline.reconfig_with_shared_info", allow_missing=False
        )
        factory_mods = reconfig_fn(factory_mods, json.dumps(shared_config))
        for idx in pipeline_mods:
            pipeline_mods[idx]["lib"].module = factory_mods[idx]
        for lib_index in pipeline_mods:
            pipeline_lib = pipeline_mods[lib_index]["lib"]
            dev = pipeline_mods[lib_index]["dev"]
            lib = graph_executor.GraphModule(pipeline_lib["default"](dev))
            # Return a module list sorted by lib_index.
            mods[lib_index] = lib.module

        return mods, json.dumps(mod_config)

    def get_pipeline_executor_module(self):
        """Get the pipeline executor module.

        Returns
        -------
        module : Module
            Common interface for pipeline executor factory Module.
        """
        if not self.module:
            graph_executors, config = self.compass_graph_executor_create(
                self.pipeline_mods, self.mods_config, self.shared_config
            )
            self.pipeline_create = get_global_func(
                "tvm.pipeline_executor.create", allow_missing=False
            )
            self.module = self.pipeline_create(graph_executors, config)
        return self.module

    def export_library(self, directory_path):
        """Export the pipeline executor into disk files.

        Parameters
        ----------
        directory_path : str
            Export the files to this directory.
        """
        if not self.pipeline_mods:
            raise RuntimeError("The pipeline executor has not been initialized.")

        cfg_file = super(CompassPipelineExecutorFactoryModule, self).export_library(directory_path)
        shared_cfg_path = os.path.join(directory_path, "shared_config")
        with open(shared_cfg_path, "w") as f:
            json.dump(self.shared_config, f)

        with open(cfg_file, "r") as f:
            cfg_dict = json.load(f)
            cfg_dict["shared_config"] = shared_cfg_path
        with open(cfg_file, "w") as f:
            json.dump(cfg_dict, f)

        return cfg_file


class ExprReplace(relay.ExprMutator):
    """Helper mutator to mutate recorded expression to var"""

    def __init__(self, convert_dict, mutate_var=False):
        super(ExprReplace, self).__init__()
        self.convert_dict = convert_dict
        self.mutate_var = mutate_var

    def visit_var(self, var):
        if self.mutate_var and var in self.convert_dict:
            return self.convert_dict[var]
        return var

    def visit_global_var(self, gvar):
        if self.mutate_var and gvar in self.convert_dict:
            return self.convert_dict[gvar]
        return gvar

    def visit_call(self, call):
        new_call = super().visit_call(call)
        if call in self.convert_dict:
            return self.convert_dict[call]
        return new_call

    def visit_tuple_getitem(self, op):
        new_getitem = super().visit_tuple_getitem(op)
        if op in self.convert_dict:
            return self.convert_dict[op]
        return new_getitem


def _gen_memo(mod):
    memo = dict()

    def fvisit(expr):
        if not isinstance(expr, (relay.Var, relay.Tuple, relay.TupleGetItem, relay.Call)):
            return
        if isinstance(expr, relay.Var):
            name = str(expr.name_hint)
            memo[name] = expr
            return
        if expr.span is not None:
            name = str(expr.span.source_name.name)
            memo[name] = expr

    relay.analysis.post_order_visit(mod["main"], fvisit)
    return memo


def module_split(mod, split_conf, mod_name="subgraph_mod"):
    """Helper to split module by split_config"""
    new_mod = tvm.IRModule(mod.functions, mod.type_definitions)
    memo = _gen_memo(new_mod)

    inputs = [memo[input_idx] for input_idx in split_conf["inputs"]]
    outputs = [memo[output_idx] for output_idx in split_conf["outputs"]]

    new_vars = []
    counter = 0
    for arg in inputs:
        if isinstance(arg, relay.Var):
            new_vars.append(arg)
        else:
            var = relay.Var(f"{mod_name}_var_{counter}", arg.checked_type)
            counter = counter + 1
            new_vars.append(var)

    if len(outputs) > 1:
        tuple_expr = relay.Tuple(outputs)
        func = tvm.IRModule.from_expr(tuple_expr)["main"]
    else:
        func = tvm.IRModule.from_expr(outputs[0])["main"]
    convert_dict = dict()
    for arg, var in zip(inputs, new_vars):
        convert_dict[arg] = var

    new_func = ExprReplace(convert_dict, True).visit(func)
    new_func = relay.Function(new_vars, new_func.body)
    new_mod = tvm.IRModule.from_expr(new_func)
    new_func = new_mod["main"]
    if len(new_func.params) != len(inputs):
        raise RuntimeError(f"split for {mod_name} config not correct, free vars still exist.")
    return relay.transform.InferType()(new_mod)


def split_to_subgraph_modules(mod, split_configs):
    """Function to split module to sub modules by split_configs

    Parameters
    ----------
    mod : IRModule
        The relay Module

    split_configs : List[Dict]
        List of dict to describe how to split the module for each sub modules
        The Dict has two keys, inputs and outputs, the value is List of int or str,
        int is the SSA index when print the module, str indicate the expression is var name

    Returns
    -------
    result : list of IRModule
        The output of the splited
    """

    new_mod = relay.transform.InferType()(mod)

    mods = []
    for idx, config in enumerate(split_configs):
        sub_mod = module_split(new_mod, config, f"sub_module{idx}")
        mods.append(sub_mod)
    return mods


class CompassPipelineConfig(object):
    """Helper wrapper for PipelineConfig"""

    def __init__(self, compass):
        if compass.ir_mod is None:
            compass.parse()
            compass.optimize()
        self.ir_mod = relay.transform.InferType()(compass.ir_mod)
        # self.compass = compass
        self.split_configs = None
        self.pipe_outputs = None
        self.outputs_producer = dict()
        self.outputs_consumers = dict()
        self.output_dir = AipuCompassConfig.get().common["output_dir"]
        self.memo = dict()
        self.mods = []
        self.mods_to_fn = dict()
        self.inner_tensor = dict()
        self.fn_to_mods = dict()
        self.calibration_paths = dict()
        self.origin_opt_cfg = dict()
        self.origin_gb_cfg = dict()
        self.pipe_config = None

    def split(self, split_configs, pipe_outputs):
        """
        Function to split module to sub modules by split_configs.
        return the configured PipelineConfig object

        Parameters
        ----------
        split_configs : List[Dict]
            List of dict to describe how to split the module for each sub modules
            The Dict has two keys, inputs and outputs, the value is List of int or str,
            int is the SSA index when print the module, str indicate the expression is var name

        pipe_outputs : List[str]
            List of output name for the pipeline output, the name is from onnx node name.
            The order in the list is the output order.

        Returns
        -------
        result : PipelineConfig
            PipelineConfig is well configured
        """
        self.pipe_outputs = pipe_outputs

        mod = self.ir_mod
        self.memo = _gen_memo(mod)

        absent_tensors = dict()
        for config in split_configs:
            for inp in config["inputs"]:
                if inp not in self.memo:
                    absent_tensors[inp] = False
            for out in config["outputs"]:
                if out not in self.memo:
                    absent_tensors[out] = False

        absent_tensors = list(absent_tensors.keys())
        msg = ",".join(absent_tensors)
        if absent_tensors:
            raise RuntimeError(f"{msg} not found in the ir_mod.")

        # sort the config
        for idx, cfg in enumerate(split_configs):
            cfg["inputs"].sort()
            cfg["outputs"].sort()
            split_configs[idx] = cfg
        self.split_configs = split_configs

        tensors = [str(param.name_hint) for param in mod["main"].params]

        for config in split_configs:
            outputs = config["outputs"]
            for out in outputs:
                if out not in tensors:
                    tensors.append(out)

        # check input output
        for idx, config in enumerate(split_configs):
            inputs = config["inputs"]
            absent_tensors = [tensor for tensor in inputs if tensor not in tensors]
            if absent_tensors:
                msg = ",".join([str(tensor) for tensor in absent_tensors])
                raise RuntimeError(
                    f"The {idx} config inputs {msg} are not found in other config outputs."
                )

        self.outputs_producer = dict()
        for idx, cfg in enumerate(split_configs):
            outputs = cfg["outputs"]
            for out in outputs:
                self.outputs_producer[out] = idx

        self.outputs_consumers = dict()
        for out in self.outputs_producer:
            self.outputs_consumers[out] = []
        self.inner_tensor.clear()
        for mod_idx, cfg in enumerate(split_configs):
            inputs = cfg["inputs"]
            for inp_idx, inp in enumerate(inputs):
                if inp in self.outputs_producer:
                    self.outputs_consumers[inp].append((mod_idx, inp_idx))
                    if inp not in self.inner_tensor and inp not in pipe_outputs:
                        self.inner_tensor[inp] = None

        out_absent = [out for out in pipe_outputs if out not in self.outputs_producer]
        if out_absent:
            msg = ",".join([str(tensor) for tensor in out_absent])
            raise RuntimeError(f"The pipe outputs {msg} are not any mdoules outputs")

        self.mods = split_to_subgraph_modules(mod, split_configs)
        # generate pipeline_config
        mods = self.mods
        pipe_config = pipeline_executor_build.PipelineConfig()

        mod_inputs = [str(param.name_hint) for param in mod["main"].params]
        for cfg_idx, config in enumerate(split_configs):
            inputs = config["inputs"]
            mod = mods[cfg_idx]
            for inp in inputs:
                if inp in mod_inputs:
                    pipe_config["input"][inp].connect(pipe_config[mod]["input"][inp])

        for cfg_idx, config in enumerate(split_configs):
            outputs = config["outputs"]
            mod = mods[cfg_idx]
            for out_idx, out in enumerate(outputs):
                if out in pipe_outputs:
                    pipe_out_idx = pipe_outputs.index(out)
                    pipe_config[mod]["output"][out_idx].connect(pipe_config["output"][pipe_out_idx])
                consumers = self.outputs_consumers[out]
                for consumer in consumers:
                    mod_idx, inp_idx = consumer
                    param_name = str(mods[mod_idx]["main"].params[inp_idx].name_hint)
                    pipe_config[mod]["output"][out_idx].connect(
                        pipe_config[mods[mod_idx]]["input"][param_name]
                    )

        self.pipe_config = pipe_config
        return pipe_config

    def prepare_build(self, target="llvm", export_cc=None):
        """Helper to set the build function for each submodule"""
        target_aipu = AipuCompassConfig.get().gbuilder["target"]
        value_from_env = os.environ.get("AIPU_TVM_GBUILDER_TARGET", None)
        if value_from_env is None:
            os.environ["AIPU_TVM_GBUILDER_TARGET"] = target_aipu

        inner_expr = dict()
        for cfg in self.split_configs:
            for inp_idx in cfg["inputs"]:
                expr = self.memo[inp_idx]
                inner_expr[inp_idx] = expr

        tuple_expr = relay.Tuple(list(inner_expr.values()))
        params = self.ir_mod["main"].params
        value_func = relay.Function(params, tuple_expr)
        value_mod = tvm.IRModule.from_expr(value_func)

        run_wrapper = relay.create_executor(kind="graph", mod=value_mod).evaluate()

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
        if AipuCompassConfig.get().common["without_batch_dim"] == "true":
            minibatch_size = 1
        else:
            minibatch_size = batch_size
        data_loader = PyTorchDataLoader(calibrate_dataset, batch_size=minibatch_size)

        out_values = []
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
            out = run_wrapper(*args)
            out_values.append(out)

        batched_args = dict()
        for idx, out_idx in enumerate(inner_expr):
            all_outs = [value[idx].numpy() for value in out_values]
            batched_arg = np.concatenate(all_outs)
            batched_args[out_idx] = batched_arg

        output_dir = AipuCompassConfig.get().common["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        self.calibration_paths = dict()
        self.mods_to_fn = dict()
        for idx, cfg in enumerate(self.split_configs):
            inputs = cfg["inputs"]
            arg_values = [batched_args[inp] for inp in inputs]
            path = os.path.join(output_dir, f"calibrateion_data{idx}")
            np.savez(path, *arg_values)
            path = path + ".npz"
            fn_name = f"submodule_{idx}"
            self.calibration_paths[fn_name] = path
            self.mods_to_fn[self.mods[idx]] = fn_name

        shared_cfg = dict()
        shared_cfg["modules"] = []
        for idx, cfg in enumerate(self.split_configs):
            mod_shared = dict()
            mod_shared["inputs"] = []
            inputs = cfg["inputs"]
            for inp in inputs:
                if inp in self.inner_tensor:
                    mod_shared["inputs"].append(inp)
                else:
                    mod_shared["inputs"].append("not_shared")

            mod_shared["outputs"] = []
            outputs = cfg["outputs"]
            for out in outputs:
                if out in self.inner_tensor:
                    mod_shared["outputs"].append(out)
                else:
                    mod_shared["outputs"].append("not_shared")
            shared_cfg["modules"].append(mod_shared)

        self.shared_config = shared_cfg
        origin_opt_cfg = dict(AipuCompassConfig.get().optimizer)
        origin_gb_cfg = dict(AipuCompassConfig.get().gbuilder)

        def _build_func(
            ir_mod,
            target,
            params=None,
            target_host=None,
            mod_name=None,
        ):  # pylint: disable=unused-argument
            fn_name = self.mods_to_fn[ir_mod]

            mod_idx = self.mods.index(ir_mod)
            split_config = self.split_configs[mod_idx]
            inputs = split_config["inputs"]
            outputs = split_config["outputs"]
            bypass_quant = []
            bypass_dequant = []
            for idx, inp in enumerate(inputs):
                if inp in self.inner_tensor:
                    bypass_quant.append(idx)
            for idx, out in enumerate(outputs):
                if out in self.inner_tensor:
                    bypass_dequant.append(idx)
            submod_output_dir = os.path.join(self.output_dir, fn_name)
            os.makedirs(submod_output_dir, exist_ok=True)

            opt_cfg = copy.copy(origin_opt_cfg)
            opt_cfg["calibration_data"] = self.calibration_paths[fn_name]
            opt_cfg["dataset"] = "numpymultiinputdataset"
            if "statistic_file" in opt_cfg:
                opt_cfg.pop("statistic_file")

            if AipuCompassConfig.get().common["compat_quantized_model"] == "true":
                opt_cfg["compat_quantized_model"] = "true"

            gb_cfg = copy.copy(origin_gb_cfg)
            if bypass_quant:
                gb_cfg["disable_input_buffer_reuse"] = "true"
            if bypass_dequant:
                gb_cfg["alloc_output_on_extsram"] = "true"

            cfg_str = textwrap.dedent(
                f"""
                [Common]
                executor = graph
                output_dir = {submod_output_dir}

                [Parser]

                [Optimizer]"""
            )
            cfg_str += "\n"

            for key in opt_cfg:
                cfg_str += f"{key} = {opt_cfg[key]}\n"

            cfg_str += "\n[GBuilder]\n"
            for key in gb_cfg:
                cfg_str += f"{key} = {gb_cfg[key]}\n"

            class _CompiledThread(Thread):
                """Build function run on another thread"""

                def __init__(self, cfg_str, ir_mod, target):
                    super(_CompiledThread, self).__init__()
                    self.cfg_str = cfg_str
                    self.ir_mod = ir_mod
                    self.target = target
                    self.result = None

                def run(self):
                    compass_cfg = AipuCompass(self.cfg_str)
                    compass_cfg.ir_mod = self.ir_mod
                    compass_cfg.optimize()
                    compass_cfg.partition()
                    ir_mod = compass_cfg.ir_mod

                    assert len(ir_mod.functions) == 2
                    gvar, main_var = ir_mod.functions.keys()
                    if main_var.name_hint != "main":
                        main_var, gvar = gvar, main_var

                    func = ir_mod.functions[gvar]
                    new_name = str(gvar.name_hint) + "_" + fn_name
                    func = func.with_attr("global_symbol", new_name)

                    new_gvar = relay.GlobalVar(new_name, gvar.checked_type)
                    var_dict = {gvar: new_gvar}
                    main_func = ExprReplace(var_dict, True).visit(ir_mod["main"])
                    ir_mod.update_func(main_var, main_func)
                    ir_mod.update_func(new_gvar, func)
                    ir_mod.remove(gvar)
                    compass_cfg.ir_mod = relay.transform.InferType()(compass_cfg.ir_mod)

                    compass_cfg.build_aipu_subgraphs(
                        bypass_input_quant=bypass_quant, bypass_output_dequant=bypass_dequant
                    )

                    deployable = compass_cfg.build(self.target)
                    self.result = deployable._compiled_model

                def get_result(self):
                    return self.result

            buildthread = _CompiledThread(cfg_str, ir_mod, target)
            buildthread.start()
            buildthread.join()

            return buildthread.result

        # build submodule
        for mod in self.mods:
            self.pipe_config[mod].target = target
            self.pipe_config[mod].dev = tvm.cpu(0)
            self.pipe_config[mod].mod_name = self.mods_to_fn[mod]
            if export_cc:
                self.pipe_config[mod].export_cc = export_cc
            self.pipe_config[mod].build_func = _build_func

    def build(self):
        """build function"""
        with tvm.transform.PassContext(opt_level=3):
            pipeline_mod_factory = pipeline_executor_build.build(self.pipe_config)
        pipeline_mods = pipeline_mod_factory.pipeline_mods
        mods_config = pipeline_mod_factory.mods_config
        return CompassPipelineExecutorFactoryModule(pipeline_mods, mods_config, self.shared_config)
