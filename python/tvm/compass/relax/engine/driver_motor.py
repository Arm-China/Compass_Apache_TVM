# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Build Compass function for running through Compass Driver."""
import time
from tvm import target as tgt
from ...logger import DEBUG
from ...runtime import CompassModule
from ...utils import check_call_compass_tool
from ..codegen import CodeGenCompass
from ..config import CompassConfig, CompassFunctionConfig
from ..utils import relative_symlink_in_dir
from .engine import ForwardEngine


def _create_compass_module(cps_bin_path, func_name):
    cfg = CompassConfig.get()
    target = cfg.gbuilder["target"]
    gb_dtcm_sz = cfg.gbuilder.get("tcm_size", None)
    # The size in GBuilder is kBytes, the size in UMD is MBytes.
    umd_dtcm_sz = str(int(gb_dtcm_sz) // 1024) if gb_dtcm_sz else ""
    with_profile = "profile" in cfg.gbuilder
    return CompassModule(cps_bin_path, func_name, with_profile, target, umd_dtcm_sz)


class DriverMotor(ForwardEngine):
    """Build Compass function for running through Compass Driver."""

    def pre_build(self, func):
        """Build the Compass function to run on Compass Driver."""
        cfg = CompassFunctionConfig(func.attrs.global_symbol)
        # 1. Get the Compass IR from Relax IR and write them to disk.
        start = time.perf_counter()
        DEBUG("Relax to Compass IR start")
        CodeGenCompass().gen2file(func, *cfg.compass_ir_path)
        DEBUG(f"Relax to Compass IR finished, elapsed time: {(time.perf_counter() - start):.2f}s")

        # 2. Simplify Compass float IR through "aipugsim" if needed.
        if cfg.use_gsim_float:
            # Create symbolic links that point to Compass IR in gsim_float directory.
            src_paths = (*cfg.compass_ir_path,)
            relative_symlink_in_dir(src_paths, cfg.gsim_float_work_dir)
            try:
                start = time.perf_counter()
                DEBUG("Compass GSim float IR start")
                check_call_compass_tool(cfg.gsim_cmd("float"), cfg.gsim_float_work_dir)
                elapsed_time = time.perf_counter() - start
                DEBUG(f"Compass GSim float IR finished, elapsed time: {elapsed_time:.2f}s")
            except:  # pylint: disable=bare-except
                cfg.gsim_float_ir_path = cfg.compass_ir_path

        # 3. Get the quantized Compass IR through "aipuopt".
        # Create symbolic links that point to Compass IR(gsim) in Optimizer directory.
        float_ir_path = cfg.gsim_float_ir_path if cfg.use_gsim_float else cfg.compass_ir_path
        src_paths = (*float_ir_path,)
        relative_symlink_in_dir(src_paths, cfg.optimizer_work_dir)
        start = time.perf_counter()
        DEBUG("Compass Optimizer start")
        check_call_compass_tool(cfg.optimizer_cmd, cfg.optimizer_work_dir)
        DEBUG(f"Compass Optimizer finished, elapsed time: {(time.perf_counter() - start):.2f}s")

        # 4. Simplify Compass quant IR through "aipugsim".
        # Create symbolic links that point to quantized Compass IR in gsim_quant directory.
        src_paths = (*cfg.quant_compass_ir_path,)
        relative_symlink_in_dir(src_paths, cfg.gsim_quant_work_dir)
        try:
            start = time.perf_counter()
            DEBUG("Compass GSim quantized IR start.")
            check_call_compass_tool(cfg.gsim_cmd("quant"), cfg.gsim_quant_work_dir)
            elapsed_time = time.perf_counter() - start
            DEBUG(f"Compass GSim quantized IR finished, elapsed time: {elapsed_time:.2f}s")
        except:  # pylint: disable=bare-except
            cfg.gsim_quant_ir_path = cfg.quant_compass_ir_path

        # 5. Get the Compass executable(i.e., aipu.bin) through "aipugb".
        # Create symbolic links that point to quantized Compass IR(gsim) in GBuilder directory.
        src_paths = (*cfg.gsim_quant_ir_path,)
        relative_symlink_in_dir(src_paths, cfg.gbuilder_work_dir)
        DEBUG("Compass GBuilder start")
        start = time.perf_counter()
        check_call_compass_tool(cfg.gbuilder_cmd, cfg.gbuilder_work_dir)
        DEBUG(f"Compass GBuilder finished, elapsed time: {(time.perf_counter() - start):.2f}s")

        return func

    def build(self, func):
        """Create the Compass runtime module with the pre-build result during Relax build the whole
        model."""
        func_name = func.attrs.global_symbol

        cfg = CompassFunctionConfig(func_name)
        cps_mod = _create_compass_module(cfg.gbuilder_output_file, func_name)
        return tgt._ffi_api.AttachCompassModuleToLLVM(cps_mod, str(func.attrs["target"]))
