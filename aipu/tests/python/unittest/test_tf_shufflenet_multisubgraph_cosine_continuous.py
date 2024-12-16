# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import os
import numpy as np
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass import AipuCompass
from tvm.relay.backend.contrib.aipu_compass.config import AipuCompassConfig
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def test_cosine_continuous():
    ZHOUYI_MODEL_ZOO_HOME = os.getenv("ZHOUYI_MODEL_ZOO_HOME")
    cfg = f"""
    [Common]
    continuous_similarity = True

    [Parser]
    model_type = tensorflow
    model_name = shufflenet_v2
    input_model = {ZHOUYI_MODEL_ZOO_HOME}/tf_shufflenet_v2/shufflenet_v2.pb
    input = Placeholder
    input_shape = [1, 224, 224, 3]
    output = classifier/BiasAdd

    [Optimizer]
    calibration_batch_size = 1
    weight_bits = 8
    bias_bits = 32
    activation_bits = 8
    dataset = NumpyDataset
    calibration_data = {ZHOUYI_MODEL_ZOO_HOME}/tf_shufflenet_v2/calibration_data.npy

    [GBuilder]
    target = X1_1204
    """
    pool = relay.op.get("nn.max_pool2d")
    pool.reset_attr("target.aipu_compass")
    mean = relay.op.get("mean")
    mean.reset_attr("target.aipu_compass")

    compass = AipuCompass(cfg)
    deployable = compass.compile()
    ee = deployable.create_execution_engine()
    image = np.load(f"{ZHOUYI_MODEL_ZOO_HOME}/tf_shufflenet_v2/calibration_data.npy")

    outputs = ee.run(image)
    output_dir = AipuCompassConfig.get().common["output_dir"]
    for folder in os.listdir(output_dir):
        if folder.startswith("tvmgen_default"):
            json_file = os.path.join(output_dir, folder, "optimizer", "opt_continuous_similarity.json")
            assert os.path.exists(json_file), f"Continuous sim json {json_file} not existed!"

    tf_model = aipu_testing.TFModel(f"{aipu_testing.DATA_DIR}/tf_shufflenet_v2.cfg")
    aipu_testing.get_test_result(tf_model, image, outputs)


if __name__ == "__main__":
    test_cosine_continuous()
