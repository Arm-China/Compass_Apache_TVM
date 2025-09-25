# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The APIs need to be exposed to other packages."""
import tvm
from .engine import ForwardEngine
from .driver_motor import DriverMotor
from .opt_forward_engine import OptForwardEngine
from .collector import Collector
from .gt_forward_engine import GtForwardEngine


def create_forward_engine(engine_type):
    """Create the engine which determine how to run the Compass function during run stage, e.g.,
    through Compass Driver, Compass Optimizer."""
    if engine_type == "driver":
        return DriverMotor()
    if engine_type == "opt_float":
        return OptForwardEngine(False)
    if engine_type == "opt_int":
        return OptForwardEngine(True)
    if engine_type == "gt":
        return GtForwardEngine()
    raise ValueError(f'Invalid Compass forward engine "{engine_type}".')


def create_calibrate_collector(collector_type):
    """Create the engine which determine how to run the Compass function during collect calibration
    data stage, e.g., through Relax VM, Compass Optimizer."""
    if collector_type in ("relax_vm", "opt"):
        return Collector(collector_type)
    raise ValueError(f'Invalid Compass calibrate collector "{collector_type}".')


@tvm.register_func("relax.ext.compass")
def _build_compass(functions, options, constant_names):  # pylint: disable=unused-argument
    ret = []
    for func in functions:
        if "compass.pre_build" in func.attrs:
            ret.append(create_forward_engine(func.attrs["compass.pre_build"]).build(func))
        else:
            msg = "Relax's build stage must be inside any kind of ForwardEngine context manager, "
            msg += "when there're Compass functions in the IRModule and they aren't be pre-built."
            assert ForwardEngine.current, msg
            ret.append(ForwardEngine.current.build(func))
    return ret
