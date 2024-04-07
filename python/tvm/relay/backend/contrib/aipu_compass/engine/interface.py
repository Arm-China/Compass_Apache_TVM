# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The APIs need to be exposed to other packages."""
import tvm
from .engine import AipuForwardEngine
from .driver_motor import CompassDriverMotor
from .opt_forward_engine import AIPUOptForwardEngine
from .collector import AipuCollector
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


def create_calibrate_collector(collector_type):
    """Create the engine which determine how to run the AIPU Compass function
    during collect calibration data stage, e.g., through Relay VM, AIPU Compass
    Optimizer."""
    if collector_type in ("relay_vm", "opt"):
        return AipuCollector(collector_type)
    raise ValueError(f'Invalid AIPU calibrate collector "{collector_type}".')


@tvm.register_func("relay.ext.aipu_compass")
def _build_aipu_compass(func):
    """The callback function for AIPU Compass function."""
    if "compass.pre_build" in func.attrs:
        return create_forward_engine(func.attrs["compass.pre_build"]).build(func)

    assert AipuForwardEngine.current, (
        "Relay's build stage must be inside any kind of AipuForwardEngine context manager, when "
        "there are AIPU Compass functions in the IRModule and they aren't be pre-built."
    )
    return AipuForwardEngine.current.build(func)


@tvm.register_func("relay.ext.aipu_compass.constant_updater")
def _constant_updater(func, func_name):  # pylint: disable=unused-argument
    """The packed function is used to tell the Relay build process which
    constants of AIPU Compass function need to be extracted and stored into top
    level runtime module. Just return empty dictionary because we will embed
    them inside our AIPUCompassModule."""
    return dict()
