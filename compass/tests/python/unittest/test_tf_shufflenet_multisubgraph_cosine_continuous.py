# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import os
import numpy as np
from tvm import relax
from tvm.compass.relax import Compass, CompassConfig, testing
from tvm.compass.relax.op.pattern_table import FLOAT_PATTERNS


def run_shufflenet():
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
    compass = Compass(cfg)
    deployable = compass.compile()
    ee = deployable.create_execution_engine()
    image = np.load(f"{ZHOUYI_MODEL_ZOO_HOME}/tf_shufflenet_v2/calibration_data.npy")

    outputs = ee.run(image)
    output_dir = CompassConfig.get().common["output_dir"]
    dirs = os.listdir(output_dir)
    num_subgraph = [x.startswith("tvm_compass_subfunc") for x in dirs].count(True)
    assert num_subgraph == 3
    for folder in dirs:
        if folder.startswith("tvm_compass_subfunc"):
            json_file = os.path.join(output_dir, folder, "optimizer", "opt_continuous_similarity.json")
            assert os.path.exists(json_file), f"Continuous sim json {json_file} not existed!"

    tf_model = testing.TFModel(f"{testing.DATA_DIR}/tf_shufflenet_v2.cfg")
    testing.get_test_result(tf_model, image, outputs)


def test_cosine_continuous():
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
        run_shufflenet()
    finally:
        FLOAT_PATTERNS = pre_table


if __name__ == "__main__":
    test_cosine_continuous()
