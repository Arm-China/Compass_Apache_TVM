# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import os
import numpy as np
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass import AipuCompass
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing

# Import TensorFlow locally in function will lead resource leak.
import tensorflow as tf

try:
    # Package "tf.compat.v1" is added from version "r1.13".
    tf_compat_v1 = tf.compat.v1  # pylint: disable=invalid-name
except AttributeError:
    tf_compat_v1 = tf  # pylint: disable=invalid-name


ZHOUYI_MODEL_ZOO_HOME = os.getenv("ZHOUYI_MODEL_ZOO_HOME")


def test_cfg_without_parser():
    cfg = f"""
    [Optimizer]
    calibration_batch_size = 20
    weight_bits = 8
    bias_bits = 32
    activation_bits = 8
    dataset = NumpyDataset
    calibration_data = {ZHOUYI_MODEL_ZOO_HOME}/tf_mobilenet_v2/calibration_data.npy

    [GBuilder]
    target = X1_1204
    """

    # 1. Parse the nn model to Relay IR.
    model_path = f"{ZHOUYI_MODEL_ZOO_HOME}/tf_mobilenet_v2/mobilenet_v2.pb"
    with open(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())

    shape_dict = {"input": (1, 224, 224, 3)}
    outputs = ["MobilenetV2/Predictions/Reshape_1:0"]
    ir_mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict, outputs=outputs)
    ir_mod["main"] = relay.build_module.bind_params_by_name(ir_mod["main"], params)

    # 2. Optimize and partition for AIPU Compass.
    #   a. Create AIPU Compass instance and set configurations.
    compass = AipuCompass(cfg)
    compass.ir_mod = ir_mod
    #   b. Optimize the nn model for AIPU Compass.
    compass.optimize()
    #   c. Partition the nn model for AIPU Compass.
    compass.partition()
    #   d. Collect calibration data for all AIPU Compass functions if needed.
    compass.collect_calibration_data()

    # 3. Compile the nn model.
    deployable = compass.build(target="llvm")

    # 4. Create execution engine.
    ee = deployable.create_execution_engine()

    preprocessed_image = aipu_testing.get_imagenet_input(preprocess_mode="tf_inc")
    outputs = ee.run(preprocessed_image)

    # 5. Check correctness for regression test.
    # Check inference result.
    predictions = outputs[0].numpy()[0]
    top3_idxes = np.argsort(predictions)[-3:][::-1]
    synset = aipu_testing.get_imagenet_synset(1001)
    print(f"Top3 labels: {[synset[idx] for idx in top3_idxes]}")
    print(f"Top3 predictions: {[predictions[idx] for idx in top3_idxes]}")

    # 6. Check cosine distance with the outputs of original framework.
    tf_model = aipu_testing.TFModel(cfg)
    tf_model.model_path = model_path
    tf_model.model_name = "mobilenet_v2"
    aipu_testing.get_test_result(tf_model, preprocessed_image, outputs)


if __name__ == "__main__":
    test_cfg_without_parser()
