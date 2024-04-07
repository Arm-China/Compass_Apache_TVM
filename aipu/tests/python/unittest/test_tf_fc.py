# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import tensorflow as tf
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def fc(inp, output_num):
    if len(inp.shape) > 2:
        inp = tf.reshape(inp, (int(inp.shape[0]), -1))

    weight = tf.Variable(
        tf.random.truncated_normal([int(inp.shape[-1]), output_num], dtype=tf.float32, stddev=1e-1),
        name="weights",
    )
    net = tf.matmul(inp, weight)

    biases = tf.compat.v1.get_variable(
        "biases", [output_num], dtype=tf.float32, initializer=tf.constant_initializer(0.1)
    )
    return tf.nn.bias_add(net, biases)


@pytest.mark.parametrize("input_shapes, out_num", [([[1, 3, 12, 12]], 10)])
def test_fc(input_shapes, out_num):
    op_type = "FC"
    model_name = aipu_testing.gen_model_name(op_type, input_shapes, out_num)

    g = tf.Graph()
    with g.as_default():
        inputs = aipu_testing.get_input_tensor_of_tf(input_shapes)
        inp = inputs[0]
        out = fc(inp, out_num)

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
