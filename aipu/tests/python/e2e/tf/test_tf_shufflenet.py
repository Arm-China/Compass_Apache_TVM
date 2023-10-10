# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
import os
import pytest
import numpy as np
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass import AipuCompass, AipuCompassConfig
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def run_shufflenet(model, runtime):
    cfg = f"{aipu_testing.DATA_DIR}/tf_{model}.cfg"
    # 1. Create AIPU Compass instance and set configurations.
    compass = AipuCompass(cfg)

    # 2. Compile the nn model.
    target = "llvm -mtriple=aarch64-linux-gnu" if runtime == "rpc" else "llvm"
    deployable = compass.compile(target=target)

    # 3. Create execution engine.
    rpc_sess, device_compiler = None, None
    if runtime == "rpc":
        rpc_sess = aipu_testing.get_rpc_session()
        device_compiler = aipu_testing.DEVICE_COMPILER
        assert device_compiler is not None, "need to set device cross compiler path."
    ee = deployable.create_execution_engine(rpc_sess, cc=device_compiler)

    # 4. Run the nn model.
    opt_cfg = AipuCompassConfig.get().optimizer
    image = np.load(f"{os.path.expandvars(opt_cfg['calibration_data'])}")[:1]
    outputs = ee.run(image)

    # 5. Check cosin distance
    tf_model = aipu_testing.TFModel(f"{aipu_testing.DATA_DIR}/tf_{model}.cfg")
    aipu_testing.get_test_result(tf_model, image, outputs, runtime=runtime)


@pytest.mark.X2_1204
@pytest.mark.parametrize("runtime", ["rpc", "simulator"])
@aipu_testing.clear_traceback
def test_shufflenet_v2(runtime):
    pool = relay.op.get("nn.max_pool2d")
    pool_check = pool.get_attr("target.aipu_compass")
    pool.reset_attr("target.aipu_compass")
    mean = relay.op.get("mean")
    mean_check = mean.get_attr("target.aipu_compass")
    mean.reset_attr("target.aipu_compass")
    try:
        run_shufflenet("shufflenet_v2", runtime)
    finally:
        pool.set_attr("target.aipu_compass", pool_check)
        mean.set_attr("target.aipu_compass", mean_check)


if __name__ == "__main__":
    test_shufflenet_v2("rpc")
