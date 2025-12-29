# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import os
import pytest
import numpy as np
from tvm.compass.relax import Compass, testing


def run_deeplab_v3(model, runtime):
    # 1. Create Compass instance and set configurations.
    cfg = f"{testing.DATA_DIR}/tf_{model}.cfg"
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
    dataset_home = os.environ["ZHOUYI_MODEL_ZOO_HOME"]
    data_file = f"{dataset_home}/tf_{model}/input.bin"
    images = np.fromfile(data_file, dtype="uint8").reshape([-1, 513, 513, 3])
    image = images[:1]
    outputs = ee.run(image)

    # 5. Check cosin distance.
    tf_model = testing.TFModel(cfg)
    testing.get_test_result(tf_model, image, outputs, runtime=runtime)


@pytest.mark.parametrize("runtime", ("rpc", "sim"))
@testing.clear_traceback
def test_deeplab_v3(runtime):
    run_deeplab_v3("deeplab_v3", runtime)


if __name__ == "__main__":
    test_deeplab_v3("sim")
