# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import os
import pytest
import numpy as np
from tvm import relax
from tvm.compass.relax import Compass, CompassConfig, testing
from tvm.compass.relax.op.pattern_table import FLOAT_PATTERNS


def run_shufflenet(model, runtime):
    cfg = f"{testing.DATA_DIR}/tf_{model}.cfg"
    # 1. Create Compass instance and set configurations.
    compass = Compass(cfg)

    # 2. Compile the nn model.
    target = "llvm -mtriple=aarch64-linux-gnu" if runtime == "rpc" else "llvm"
    deployable = compass.compile(target=target)

    # 3. Create execution engine.
    rpc_sess, device_compiler = None, None
    if runtime == "rpc":
        rpc_sess = testing.get_rpc_session()
        device_compiler = testing.DEVICE_COMPILER
        assert device_compiler is not None, "need to set device cross compiler path."
    ee = deployable.create_execution_engine(rpc_sess, cc=device_compiler)

    # 4. Run the nn model.
    opt_cfg = CompassConfig.get().optimizer
    image = np.load(f"{os.path.expandvars(opt_cfg['calibration_data'])}")[:1]
    outputs = ee.run(image)

    # 5. Check cosin distance
    tf_model = testing.TFModel(f"{testing.DATA_DIR}/tf_{model}.cfg")
    testing.get_test_result(tf_model, image, outputs, runtime=runtime)


@pytest.mark.parametrize("runtime", ("rpc", "sim"))
@testing.clear_traceback
def test_shufflenet_v2(runtime):
    global FLOAT_PATTERNS
    pre_table = FLOAT_PATTERNS[:]
    for pattern in FLOAT_PATTERNS[:]:
        if not isinstance(pattern[1], relax.dpl.CallPattern):
            continue
        if not isinstance(pattern[1].op, relax.dpl.ExprPattern):
            continue
        op = pattern[1].op.expr
        if op.name in ["relax.nn.max_pool2d", "relax.mean"]:
            FLOAT_PATTERNS.remove(pattern)
    try:
        run_shufflenet("shufflenet_v2", runtime)
    finally:
        FLOAT_PATTERNS = pre_table


if __name__ == "__main__":
    test_shufflenet_v2("sim")
