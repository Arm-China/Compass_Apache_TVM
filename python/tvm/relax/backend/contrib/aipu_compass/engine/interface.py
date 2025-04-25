# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The APIs need to be exposed to other packages."""
import tvm
from .engine import AipuForwardEngine
from .driver_motor import CompassDriverMotor
from .opt_forward_engine import AIPUOptForwardEngine

# from .collector import AipuCollector
from .gt_forward_engine import AIPUGtForwardEngine


def create_forward_engine(engine_type):
    """Create the engine which determine how to run the AIPU Compass function
    during run stage, e.g., through AIPU Driver, Compass Optimizer."""
    if engine_type == "driver":
        return CompassDriverMotor()
    if engine_type == "opt_float":
        return AIPUOptForwardEngine(False)
    if engine_type == "opt_int":
        return AIPUOptForwardEngine(True)
    if engine_type == "gt":
        return AIPUGtForwardEngine()
    raise ValueError(f'Invalid AIPU forward engine "{engine_type}".')


# todo
# def create_calibrate_collector(collector_type):
#     """Create the engine which determine how to run the AIPU Compass function
#     during collect calibration data stage, e.g., through Relax VM, AIPU Compass
#     Optimizer."""
#     if collector_type in ("relax_vm", "opt"):
#         return AipuCollector(collector_type)
#     raise ValueError(f'Invalid AIPU calibrate collector "{collector_type}".')


@tvm.register_func("relax.ext.aipu_compass")
def _build_aipu_compass(functions, options, constant_names):  # pylint: disable=unused-argument
    ret = []
    for func in functions:
        if "compass.pre_build" in func.attrs:
            ret.append(create_forward_engine(func.attrs["compass.pre_build"]).build(func))
        else:
            ret.append(AipuForwardEngine.current.build(func))
    return ret
