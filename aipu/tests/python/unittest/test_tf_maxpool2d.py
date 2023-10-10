# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
import pytest
import tensorflow as tf
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def get_tf_ver():
    _ = tf.__version__
    ver = _.split(".")[:2]
    return float(".".join(ver))


c_tf_version = get_tf_ver()


@pytest.mark.parametrize(
    "input_shapes, ksize, strides, padding",
    [
        ([[1, 224, 224, 3]], [1, 7, 11, 1], [1, 6, 5, 1], "SAME"),
        ([[1, 224, 224, 3]], [1, 7, 11, 1], [1, 5, 6, 1], "VALID"),
        ([[1, 224, 224, 3]], [1, 11, 7, 1], [1, 6, 5, 1], "SAME"),
        ([[1, 224, 224, 3]], [1, 11, 7, 1], [1, 5, 6, 1], "VALID"),
        ([[1, 224, 224, 3]], [1, 7, 7, 1], [1, 3, 3, 1], "SAME"),
        ([[1, 224, 224, 3]], [1, 7, 7, 1], [1, 3, 3, 1], "VALID"),
    ],
)
def test_maxpool2d(input_shapes, ksize, strides, padding):
    op_type = "MaxPool2D"
    model_name = aipu_testing.gen_model_name(op_type, input_shapes, ksize, strides, padding)

    g = tf.Graph()
    with g.as_default():
        inputs = aipu_testing.get_input_tensor_of_tf(input_shapes)
        inp = inputs[0]
        if c_tf_version < 1.14:
            out = tf.nn.max_pool(inp, ksize=ksize, strides=strides, padding=padding)
        else:
            out = tf.nn.max_pool2d(inp, ksize=ksize, strides=strides, padding=padding)

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shapes,
        "inputs": inputs,
        "outputs": [out],
        "in_graph": g,
    }

    cfg_file = aipu_testing.get_model_cfg_path(model_info, "tf")
    input_data = aipu_testing.get_op_input(model_name, input_shapes)
    aipu_output = aipu_testing.get_tvm_output(cfg_file, input_data)
    aipu_testing.get_test_result(aipu_testing.TFModel(cfg_file), input_data, aipu_output, 0.99)
