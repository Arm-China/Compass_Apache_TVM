# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Configuration processing of AIPU Compass."""
import os
import uuid
import shutil
from configparser import ConfigParser
import json
import tvm
from tvm.aipu.logger import WARN
from . import _ffi_api


# The type key string here must be identical with the value of C++ variable
# "AipuCompassConfigObj::_type_key".
@tvm.register_object("aipu_compass.AipuCompassConfig")
class AipuCompassConfig(tvm.Object):
    """The singleton AIPU Compass configuration.

    Attributes
    ----------
    common : dict
        The dictionary corresponding to the Common section of the configuration file.

    parser : dict
        The dictionary corresponding to the Parser section of the configuration file.

    optimizer : dict
        The dictionary corresponding to the Optimizer section of the configuration file.

    gbuilder : dict
        The dictionary corresponding to the GBuilder section of the configuration file.
    """

    @staticmethod
    def init_singleton(common, parser, optimizer, gbuilder, runtime):
        _ffi_api.AipuCompassConfig_InitSingleton(common, parser, optimizer, gbuilder, runtime)

    @staticmethod
    def get():
        """The static function that is used to get the global
        Zhouyi NPU Compass configuration singleton.

        Returns
        -------
        result : AipuCompassConfig
            The global Zhouyi NPU Compass configuration singleton.
        """
        return _ffi_api.AipuCompassConfig_Global()

    @property
    def deploy_file(self):
        framework = self.parser.get("model_type", "compiled")
        model_name = self.parser.get("model_name", "model")
        filename = f'{self.common["executor"]}_{framework}_{model_name}.so'
        return os.path.join(self.common["output_dir"], filename)

    @property
    def optimizer(self):
        return self.optimizers["0"]  # Get the first Optimizer section of the configuration.


class AipuCompassFunctionConfig:
    """Collection of various information used to process an AIPU Compass
    function through AIPU Compass tools."""

    _IR_NAME = "nn_model"
    _QUANT_IR_NAME = "quant_nn_model"

    def __init__(self, func_name):
        self.target = AipuCompassConfig.get().gbuilder["target"]
        self.func_name = func_name
        refined_name = func_name.replace("@", "_")
        self.output_dir = f'{AipuCompassConfig.get().common["output_dir"]}/{refined_name}'
        self.optimizer_work_dir = f"{self.output_dir}/optimizer"
        self.calibration_data = f"{self.optimizer_work_dir}/calibration_data.npz"
        self.optimizer_config_file = f"{self.optimizer_work_dir}/aipuopt.cfg"
        self.gbuilder_work_dir = f"{self.output_dir}/gbuilder"
        self.gbuilder_output_file = f"{self.gbuilder_work_dir}/aipu.bin"
        self.runtime_work_dir = f"{self.output_dir}/runtime"
        self.auto_fuse_dir = f'{AipuCompassConfig.get().common["output_dir"]}/auto_fuse'
        self.auto_fuse_plugin_dir = f"{self.auto_fuse_dir}/plugin"
        self.auto_fuse_lib_dir = f"{self.auto_fuse_dir}/op_lib"

        path_base = f"{self.output_dir}/{self._IR_NAME}"
        self.compass_ir_path = (f"{path_base}.txt", f"{path_base}.bin")
        path_base = f"{self.optimizer_work_dir}/{self._QUANT_IR_NAME}"
        self.quant_compass_ir_path = (f"{path_base}.txt", f"{path_base}.bin")

        self.use_gsim_float = True
        self.gsim_options = ""
        if AipuCompassConfig.get().common["simplify"].upper() in ["FALSE", "0"]:
            self.use_gsim_float = False
        else:
            self.gsim_options = AipuCompassConfig.get().common["simplify"].strip().split(" ")
        if AipuCompassConfig.get().common["compat_quantized_model"] == "true":
            self.use_gsim_float = False
        self.gsim_float_work_dir = f"{self.output_dir}/gsim_float"
        self.gsim_quant_work_dir = f"{self.output_dir}/gsim_quant"
        path_base = f"{self.gsim_float_work_dir}/{self._IR_NAME}_s"
        self.gsim_float_ir_path = (f"{path_base}.txt", f"{path_base}.bin")
        path_base = f"{self.gsim_quant_work_dir}/{self._QUANT_IR_NAME}_s"
        self.gsim_quant_ir_path = (f"{path_base}.txt", f"{path_base}.bin")

    def gen_optimizer_config_file(self, extra_cfg=None):
        """Generate the configuration file used by "aipuopt"."""
        optimizers = AipuCompassConfig.get().optimizers
        idx_str = self.func_name.split("_")[-1]
        opt_cfg = dict(optimizers[idx_str if idx_str in optimizers else "0"])
        # Generate the configuration file of AIPU optimizer and write to disk.
        ir_path = self.gsim_float_ir_path if self.use_gsim_float else self.compass_ir_path
        opt_cfg["graph"], opt_cfg["bin"] = tuple(os.path.basename(x) for x in ir_path)
        opt_cfg["output_dir"] = "."
        opt_cfg["quant_ir_name"] = self._QUANT_IR_NAME
        opt_cfg["model_name"] = self.func_name
        if os.path.isfile(self.calibration_data):
            opt_cfg["calibration_data"] = os.path.basename(self.calibration_data)
            opt_cfg["dataset"] = "numpymultiinputdataset"
        if AipuCompassConfig.get().common["compat_quantized_model"] == "true":
            opt_cfg["compat_quantized_model"] = "true"
        if "min_compatible_zhouyi_target" not in opt_cfg:
            opt_cfg["min_compatible_zhouyi_target"] = self.target.split("_")[0].strip().upper()

        if extra_cfg:
            opt_cfg.update(extra_cfg)

        cfg_parser = ConfigParser()
        cfg_parser["Common"] = opt_cfg
        os.makedirs(self.optimizer_work_dir, exist_ok=True)
        with open(self.optimizer_config_file, "w") as f:
            cfg_parser.write(f)

    @property
    def optimizer_cmd(self):
        """Generate the command line used to invoke "aipuopt"."""
        if not os.path.isfile(self.optimizer_config_file):
            self.gen_optimizer_config_file()
        return ["aipuopt", "--cfg", os.path.basename(self.optimizer_config_file)]

    @property
    def gbuilder_cmd(self):
        q_txt_path, q_bin_path = tuple(os.path.basename(x) for x in self.gsim_quant_ir_path)
        cmd = ["aipugb", q_txt_path, "-w", q_bin_path]
        for k, v in AipuCompassConfig.get().gbuilder.items():
            cmd.append(f"--{k}")
            if v != "":
                cmd.append(v)
        return cmd

    def gsim_cmd(self, stage="quant"):
        """Generate the command line used to invoke "aipugsim"."""
        if stage == "float":
            inp_txt_path, inp_bin_path = tuple(os.path.basename(x) for x in self.compass_ir_path)
            out_txt_path, out_bin_path = tuple(os.path.basename(x) for x in self.gsim_float_ir_path)
        else:
            inp_txt_path, inp_bin_path = tuple(
                os.path.basename(x) for x in self.quant_compass_ir_path
            )
            out_txt_path, out_bin_path = tuple(os.path.basename(x) for x in self.gsim_quant_ir_path)
        cmd = ["aipugsim", inp_txt_path, "-w", inp_bin_path, "-o", out_txt_path, "-b", out_bin_path]
        if stage == "float":
            cmd.append("--full")
        self.gsim_options = [option.strip() for option in self.gsim_options if len(option) > 0]
        cmd += self.gsim_options
        return cmd


def _get_simulator_path(target_major_mark):
    # Prefer the AIPU Simulator in environment variable "PATH", because the AIPU
    # Simulator record in the AIPUBuilder user configuration file maybe an old
    # and invalid one.
    for path_dir in os.get_exec_path():
        sim_path = os.path.join(path_dir, f"aipu_simulator_{target_major_mark.lower()}")
        if os.path.isfile(sim_path) and os.access(sim_path, os.X_OK):
            return sim_path

    aipu_builder_cfg_path = f'{os.path.expanduser("~")}/.aipubuilder/config'
    if os.path.isfile(aipu_builder_cfg_path):
        with open(aipu_builder_cfg_path) as f:
            aipu_builder_cfg = json.loads(f.read())
        if "simulator" in aipu_builder_cfg:
            sim_cfg = aipu_builder_cfg["simulator"]
            if target_major_mark == "Z1" and "ZYV1" in sim_cfg:
                return sim_cfg["ZYV1"]
            if target_major_mark == "Z2" and "ZYV2" in sim_cfg:
                return sim_cfg["ZYV2"]
            if target_major_mark == "Z3" and "ZYV3" in sim_cfg:
                return sim_cfg["ZYV3"]
            if target_major_mark == "X1" and "ZYX1" in sim_cfg:
                return sim_cfg["ZYX1"]

    raise RuntimeError("Can't find AIPU Simulator.")


def _check_cfg(common, parser, optimizers, gbuilder):
    def check(items, keys, name):
        for k in keys:
            if k in items:
                WARN(f"{k} is useless config item in {name} section.")

    check(["mode", "use_aqt"], common.keys(), "Common")
    check(["detection_postprocess", "model_domain", "output_dir"], parser.keys(), "Parser")

    useless_items = ("graph", "bin", "quant_precision", "quant_ir_name", "model_name", "output_dir")
    for idx_str, optimizer in optimizers.items():
        check(useless_items, optimizer.keys(), f"Optimizer_{idx_str}")

    check(["random-init", "inputs", "outputs"], gbuilder.keys(), "Gbuilder")


def _update_and_check_parser(parser, cfg_dir):
    if not parser:
        return parser
    parser.setdefault("model_type", "tensorflow")
    for key in ("input_model", "caffe_prototxt"):
        if key in parser:
            value = parser[key]
            value = value.replace("__CURRENT_CFG_DIR__", cfg_dir) if cfg_dir else value
            parser[key] = os.path.abspath(os.path.expanduser(os.path.expandvars(value)))
    return parser


def _dump_tensors(func_name, is_input, *args):
    assert args, "Args is empty which means it can't be dumped."
    cfg = AipuCompassFunctionConfig(func_name)
    dump_name = "input" if is_input else "output"
    for i, data in enumerate(args):
        data.numpy().tofile(f"{cfg.gbuilder_work_dir}/{dump_name}{i}_{data.dtype}.bin")


def config_aipu_compass(config):
    """Initialize the AIPU Compass configuration singleton using the information
    from AIPUBuilder configuration file.
    """
    cfg_str = config
    cfg_path = None
    if config.split(".")[-1] == "cfg":
        if not os.path.exists(config):
            raise FileNotFoundError(f"{config} does not exist.")
        cfg_path = os.path.dirname(os.path.abspath(config))
        with open(config) as f:
            cfg_str = f.read()

    cfg_parser = ConfigParser()
    cfg_parser.read_string(cfg_str)

    # 1. Get each section from configuration file as dictionary.
    common = dict(cfg_parser["Common"]) if "Common" in cfg_parser else dict()
    parser = dict(cfg_parser["Parser"]) if "Parser" in cfg_parser else dict()

    optimizers = {"0": {}}
    for name, section in cfg_parser.items():
        if name.startswith("Optimizer") or name.startswith("AutoQuantizationTool"):
            optimizers[name.split("_")[1] if "_" in name else "0"] = dict(section)

    gbuilder = dict(cfg_parser["GBuilder"]) if "GBuilder" in cfg_parser else dict()
    runtime = dict(cfg_parser["Runtime"]) if "Runtime" in cfg_parser else dict()
    # Lower the keys of all sections.
    for section in (common, parser, gbuilder, runtime) + tuple(optimizers.values()):
        for k, v in tuple(section.items()):
            section.pop(k)
            section[k.lower()] = v

    # 2. check all section for useless config items or not configured.
    _check_cfg(common, parser, optimizers, gbuilder)

    # 3. Update and check "common" section.
    if os.path.isfile(config):
        framework_model_name = config.split("/")[-1].split(".")[0]
        common.setdefault(
            "output_dir", f"./compass_output_{framework_model_name}_{uuid.uuid4().hex}"
        )
    else:
        common.setdefault("output_dir", f"./compass_output_{uuid.uuid4().hex}")
    common["output_dir"] = os.path.abspath(common["output_dir"])
    if os.path.exists(common["output_dir"]):
        shutil.rmtree(common["output_dir"])

    value = common.get("forward_engine", None)
    value_from_env = os.environ.get("AIPU_TVM_FORWARD_ENGINE", None)
    if value_from_env:
        value = value_from_env
    common["forward_engine"] = value.lower() if value else "driver"

    value = common.get("continuous_similarity", None)
    value_from_env = os.environ.get("AIPU_TVM_CONTINUOUS_SIM", None)
    if value_from_env:
        value = value_from_env
    common["continuous_similarity"] = value.lower() if value else "false"
    continuous_sim = common["continuous_similarity"] == "true"
    if continuous_sim:
        WARN(
            "The forward_engine has been changed to opt_int mode, "
            "because continuous sim requires every layer results inferred by opt_int."
        )
        common["forward_engine"] = "opt_int"

    value = common.get("calibrate_collector", None)
    value_from_env = os.environ.get("AIPU_TVM_CALIBRATE_COLLECTOR", None)
    if value_from_env:
        value = value_from_env
    common["calibrate_collector"] = value.lower() if value else "relay_vm"

    value = common.get("executor", None)
    value_from_env = os.environ.get("AIPU_TVM_EXECUTOR", None)
    if value_from_env:
        value = value_from_env
    common["executor"] = value.lower() if value else "graph"

    value = common.get("disabled_pass", None)
    value_from_env = os.environ.get("AIPU_TVM_DISABLED_PASS", None)
    if value_from_env:
        value = value_from_env
    common["disabled_pass"] = value or "ForwardFoldScaleAxis"

    value = common.get("log_level", None)
    value_from_env = os.environ.get("AIPU_TVM_LOG_LEVEL", None)
    if value_from_env:
        value = value_from_env
    common["log_level"] = value.upper() if value else "INFO"

    value = common.get("log_file", None)
    value_from_env = os.environ.get("AIPU_TVM_LOG_FILE", None)
    if value_from_env:
        value = value_from_env
    common["log_file"] = value if value else ""

    value = common.get("bare_metal", None)
    value_from_env = os.environ.get("AIPU_TVM_BARE_METAL", None)
    if value_from_env:
        value = value_from_env
    common["bare_metal"] = value.lower() if value else "false"

    value = common.get("disable_op_spec_checker", None)
    value_from_env = os.environ.get("AIPU_TVM_DISABLE_OP_SPEC_CHECKER", None)
    if value_from_env:
        value = value_from_env
    common["disable_op_spec_checker"] = value.lower() if value else "false"
    if common["disable_op_spec_checker"] == "true":
        WARN(
            "The Compass OP Spec checkers are disabled, please notice the "
            "generated Compass IR maybe invalid."
        )

    value = common.get("compat_quantized_model", None)
    if not value:
        value = parser.get("compat_quantized_model", None)
    value_from_env = os.environ.get("AIPU_TVM_COMPAT_QUANTIZED_MODEL", None)
    if value_from_env:
        value = value_from_env
    common["compat_quantized_model"] = value.lower() if value else "false"

    value = common.get("without_batch_dim", None)
    if not value:
        value = optimizers["0"].get("without_batch_dim", None)
    value_from_env = os.environ.get("AIPU_TVM_WITHOUT_BATCH_DIM", None)
    if value_from_env:
        value = value_from_env
    common["without_batch_dim"] = value.lower() if value else "false"

    value = common.get("compute_threshold", None)
    value_from_env = os.environ.get("AIPU_TVM_COMPUTE_THRESHOLD", None)
    if value_from_env:
        value = value_from_env
    common["compute_threshold"] = value if value else "2e5"

    value = common.get("dump_partitioning_graph", None)
    value_from_env = os.environ.get("AIPU_TVM_DUMP_PARTITIONING_GRAPH", None)
    if value_from_env:
        value = value_from_env
    common["dump_partitioning_graph"] = value.lower() if value else "false"

    value = common.get("dump_annotation_graph", None)
    value_from_env = os.environ.get("AIPU_TVM_DUMP_ANNOTATION_GRAPH", None)
    if value_from_env:
        value = value_from_env
    common["dump_annotation_graph"] = value.lower() if value else "false"

    value = common.get("dump_building_graph", None)
    value_from_env = os.environ.get("AIPU_TVM_DUMP_BUILDING_GRAPH", None)
    if value_from_env:
        value = value_from_env
    common["dump_building_graph"] = value.lower() if value else "false"

    value = common.get("auto_fuse_ops", None)
    value_from_env = os.environ.get("AIPU_TVM_AUTO_FUSE_OPS", None)
    if value_from_env:
        value = value_from_env
    common["auto_fuse_ops"] = value.lower() if value else "false"

    value = common.get("simplify", None)
    value_from_env = os.environ.get("AIPU_TVM_SIMPLIFY", None)
    if value_from_env:
        value = value_from_env
    common["simplify"] = value if value else ""

    # 4. Update and check "parser" section.
    parser = _update_and_check_parser(parser, cfg_path)

    # 5. Update and check "optimizer" section.
    removed_params = ["quant_precision"]
    for _, optimizer in optimizers.items():
        for key in removed_params:
            optimizer.pop(key, None)
        for key in ("calibration_data", "opt_config", "statistic_file", "data", "label"):
            if key in optimizer:
                new_value = os.path.abspath(os.path.expanduser(os.path.expandvars(optimizer[key])))
                optimizer[key] = new_value
        value_from_env = os.environ.get("AIPU_TVM_OPTIMIZER_SAVE_STATISTIC_INFO", None)
        if value_from_env:
            optimizer["save_statistic_info"] = value_from_env
        optimizer.setdefault("cast_dtypes_for_lib", "true")
        if continuous_sim:
            WARN(
                "The optimizer dump has been changed to true, "
                "because continuous sim requires opt dump float results."
            )
            optimizer["dump"] = "true"

    # 6. Update and check "gbuilder" section.
    value = gbuilder.get("target", None)
    value_from_env = os.environ.get("AIPU_TVM_GBUILDER_TARGET", None)
    if value_from_env:
        value = value_from_env
    gbuilder["target"] = value.upper() if value else "X1_1204"

    value = gbuilder.get("profile", None)
    value_from_env = os.environ.get("AIPU_TVM_GBUILDER_PROFILE", None)
    if value_from_env:
        value = value_from_env
    gbuilder["profile"] = value.lower() if value else "false"

    value = gbuilder.get("tcm_size", None)
    value_from_env = os.environ.get("AIPU_TVM_GBUILDER_TCM_SIZE", None)
    if value_from_env:
        value = value_from_env
    if value:
        gbuilder["tcm_size"] = value.lower()

    sim_from_gb = gbuilder.get("simulator", None)

    for k, v in gbuilder.copy().items():
        lower_v = v.lower()
        if k in ("simulator", "random-init", "inputs", "outputs"):
            gbuilder.pop(k)
        elif k == "local_lib":
            gbuilder.pop(k)
            gbuilder["Lib"] = os.path.abspath(os.path.expanduser(os.path.expandvars(v)))
        elif k == "dumpir":
            gbuilder.pop(k)
            gbuilder["dumpIR"] = v
        elif k == "debug":
            gbuilder.pop(k)
            if lower_v == "true":
                gbuilder["DEBUG"] = ""
        elif k == "prof_unit":
            gbuilder.pop(k)
            if lower_v in ("aiff", "tpc", "dma"):
                gbuilder[k] = v.upper()
        elif k in ("dump", "profile", "disable_mmu"):
            gbuilder.pop(k)
            if lower_v == "true":
                gbuilder[k] = ""
        elif lower_v == "true":
            gbuilder[k] = ""
        elif lower_v == "false":
            gbuilder.pop(k)

    # 7. Update and check "runtime" section.
    runtime.setdefault("verbose", "false")

    value = runtime.get("dump", None)
    value_from_env = os.environ.get("AIPU_TVM_RUNTIME_DUMP", None)
    if value_from_env:
        value = value_from_env
    runtime["dump"] = value.lower() if value else "false"

    if runtime["dump"] == "true":
        tvm.register_func("aipu_compass.dump_tensors", _dump_tensors, True)

    if "simulator" not in runtime:
        # For backward compatible, get the value from GBuilder section.
        runtime["simulator"] = sim_from_gb or _get_simulator_path(gbuilder["target"].split("_")[0])

    # 8. Initialize the singleton AIPU Compass configuration.
    AipuCompassConfig.init_singleton(common, parser, optimizers, gbuilder, runtime)
