# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Execute AIPU compass function through AIPU Optimizer."""
import os
import json
import numpy as np
from tvm import nd
from tvm.aipu.utils import check_call_aipu_tool
from AIPUBuilder.Optimizer.utils import dtype2nptype
from .engine import PurePythonForwardEngine, FunctionData
from ..aipu_builder import OptForward
from ..codegen import CodeGenAipuCompass
from ..config import AipuCompassConfig, AipuCompassFunctionConfig
from ..utils import relative_symlink_in_dir, compute_cos_distance


def _add_json_similarities(file_path, similarities):
    """
    add continuous similarity to another json file.
    :param file_path: the old optimizer json file path
    :param similarities: the similarities of all layers and all outputs
    :return: None
    """

    with open(file_path, "r") as file:
        data = json.load(file)
    for layer_name, cos_list in similarities.items():
        if layer_name in data:
            if "brief_info" in data[layer_name]["just_for_display"]:
                data[layer_name]["just_for_display"]["continuous_similarity"] = cos_list
        else:
            print(f"Layer {layer_name} not found in the json.")
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    save_file_path = os.path.join(directory, "opt_continuous_similarity.json")
    with open(save_file_path, "w") as file:
        json.dump(data, file, indent=4)


def _continuous_similarity_statistic(func_data):
    """
    Get the forward_engine=opt_int result and save it at the runtime folder.

    :param func_data: function_data
    :return: None
    """
    cfg = AipuCompassFunctionConfig(func_data.name)
    save_path = cfg.runtime_work_dir + "/" + "quant"
    os.makedirs(save_path, exist_ok=True)

    # Calc the cos sim between opt_int result and opt_dump float result.
    dump_dir = "opt_dump"
    opt_config = AipuCompassConfig.get().optimizer
    if "dump_dir" in opt_config:
        dump_dir = opt_config["dump_dir"]
    path_float = cfg.optimizer_work_dir + "/" + dump_dir + "/" + "float32"
    layers_similarities = {}
    nodes = func_data.executor.optimizer.g.nodes
    for n in nodes:
        for out_i, out in enumerate(n.outputs):
            # save the opt_int inference result at runtime dir
            data_quant = out.betensor.cpu().numpy().astype(dtype2nptype(out.dtype))
            np.save(
                f"{cfg.runtime_work_dir}/quant/{n.attrs['layer_id']}_opt_{out_i}_{out.name}.npy",
                data_quant,
            )
            data_float = (
                f"{func_data.name}_{n.type.split('.')[-1]}_"
                f"[{n.name}]_o{out_i}_{n.outputs[out_i].name}.npy"
            )

            # calc the cos sim
            cos = compute_cos_distance(
                np.load(path_float + "/" + data_float).flatten(),
                data_quant.flatten(),
                precision_dtype="float64",
                keep_decimal=16,
                judge_divisor_0=True,
            )

            # Append cosine similarity to the corresponding layer key
            if n.name in layers_similarities:
                layers_similarities[n.name].append(cos)
            else:
                layers_similarities[n.name] = [cos]
        json_file = cfg.optimizer_work_dir + "/" + str(func_data.name) + "_opt_template.json"
        _add_json_similarities(json_file, layers_similarities)


class AIPUOptForwardEngine(PurePythonForwardEngine):
    """Build AIPU Compass function for running through AIPU Optimizer."""

    def __init__(self, is_quant):
        super().__init__()
        self._is_quant = is_quant

    def run(self, func_data, args):
        """Responsible for executing AIPU Compass function during Relay run the
        whole compiled model."""
        np_args = [x.numpy() for x in args]
        continuous_sim = AipuCompassConfig.get().common["continuous_similarity"] == "true"

        if self._is_quant:
            ret = func_data.executor.forward_with_quantized_data(
                np_args, False, keep_tensors=continuous_sim
            )

            # Dump all quantized input and output data for debugging with tool like "aipurun".
            if AipuCompassConfig.get().runtime["dump"] == "true":
                cfg = AipuCompassFunctionConfig(func_data.name)
                os.makedirs(cfg.gbuilder_work_dir, exist_ok=True)

                for i, np_arg in enumerate(np_args):
                    np_arg.tofile(f"{cfg.gbuilder_work_dir}/input{i}_{np_arg.dtype}.bin")
                for i, np_ret in enumerate(ret):
                    np_ret.tofile(f"{cfg.gbuilder_work_dir}/output{i}_{np_ret.dtype}.bin")
        else:
            ret = func_data.executor.forward(np_args, False)

        # Check whether continuous cosine similarity statistics need to be performed.
        if continuous_sim:
            _continuous_similarity_statistic(func_data)

        return ret

    def pre_build(self, func):
        """Build the AIPU Compass function to run on AIPU Optimizer before
        Relay build the whole model."""
        # Get the Compass IR from Relay IR.
        cfg = AipuCompassFunctionConfig(func.attrs.global_symbol)
        CodeGenAipuCompass().gen2file(func, *cfg.compass_ir_path)
        ir_path = cfg.compass_ir_path

        # Simplify Compass float IR through "aipugsim" if needed.
        if cfg.use_gsim_float:
            # Create symbolic links that point to Compass IR in AIPU gsim_float directory.
            relative_symlink_in_dir(cfg.compass_ir_path, cfg.gsim_float_work_dir)
            check_call_aipu_tool(cfg.gsim_cmd("float"), cfg.gsim_float_work_dir)
            ir_path = cfg.gsim_float_ir_path

        # Get the quantized Compass IR through AIPU Optimizer if needed.
        if self._is_quant:
            # Create symbolic links that point to Compass IR in AIPU Optimizer directory.
            float_ir_path = cfg.gsim_float_ir_path if cfg.use_gsim_float else cfg.compass_ir_path
            relative_symlink_in_dir(float_ir_path, cfg.optimizer_work_dir)
            check_call_aipu_tool(cfg.optimizer_cmd, cfg.optimizer_work_dir)
            ir_path = cfg.quant_compass_ir_path

        # Read back the Compass IR.
        txt_path, bin_path = ir_path
        ir_txt = open(txt_path, "r", encoding="utf-8").read()
        ir_bin = np.fromfile(bin_path, dtype="uint8")

        try:
            # Try to obtain the NDArray object without memory copy.
            ir_bin = nd.from_dlpack(ir_bin)
        except:  # pylint: disable=bare-except
            ir_bin = nd.array(ir_bin)

        # Embed the pre-build result into Relay IR through attribute of function.
        new_attrs = {"compass.pre_build.ir_txt": ir_txt, "compass.pre_build.ir_bin": ir_bin}
        return func.with_attr(new_attrs)

    def create_func_data(self, func):
        """Responsible for creating object used to store data of the Relay
        function with the pre-build result during Relay build the whole model."""
        func_data = FunctionData(func)

        # Write the Compass IR to disk for AIPU Optimizer forward interface.
        cfg = AipuCompassFunctionConfig(func_data.name)
        txt_path, bin_path = cfg.compass_ir_path
        if self._is_quant:
            txt_path, bin_path = cfg.quant_compass_ir_path

        # Get the pre-build result back from Relay IR.
        # There is a bug in Relay parser, it won't resume the escaped characters.
        ir_txt = func.attrs["compass.pre_build.ir_txt"].encode("utf-8").decode("unicode_escape")
        ir_bin = func.attrs["compass.pre_build.ir_bin"].numpy()

        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        open(txt_path, "w", encoding="utf-8").write(ir_txt)
        os.makedirs(os.path.dirname(bin_path), exist_ok=True)
        ir_bin.tofile(bin_path)

        # Create the AIPU Optimizer forward instance.
        func_data.executor = OptForward(txt_path, bin_path)
        return func_data
