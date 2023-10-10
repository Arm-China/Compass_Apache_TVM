# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
import pytest
import tensorflow as tf
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def biasadd(conv):
    b = conv.shape[-1]
    biases = tf.compat.v1.get_variable(
        "biases", [b], dtype=tf.float32, initializer=tf.constant_initializer(0.1)
    )
    return tf.nn.bias_add(conv, biases)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 20, 30, 40]],
    ],
)
def test_biasadd(input_shapes):
    op_type = "BiasAdd"
    model_name = aipu_testing.gen_model_name(op_type, aipu_testing.gen_dim_info(input_shapes))

    g = tf.Graph()
    with g.as_default():
        inputs = aipu_testing.get_input_tensor_of_tf(input_shapes)
        inp = inputs[0]
        out = biasadd(inp)

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
