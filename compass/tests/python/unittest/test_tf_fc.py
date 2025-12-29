# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
from tvm.compass.relax import testing


@pytest.mark.parametrize("input_shapes, out_num", [([[1, 3, 12, 12]], 10)])
def test_fc(input_shapes, out_num):
    op_type = "FC"
    model_name = testing.gen_model_name(op_type, input_shapes, out_num)

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shapes,
    }

    if not testing.is_model_file_exists(op_type, "tf", model_name):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        g = tf.Graph()
        with g.as_default():
            inputs = testing.get_input_tensor_of_tf(input_shapes)
            inp = inputs[0]
            if len(inp.shape) > 2:
                inp = tf.reshape(inp, (int(inp.shape[0]), -1))
            weight = tf.Variable(
                tf.random.truncated_normal([int(inp.shape[-1]), out_num], dtype=tf.float32, stddev=1e-1),
                name="weights",
            )
            net = tf.matmul(inp, weight)
            biases = tf.compat.v1.get_variable(
                "biases", [out_num], dtype=tf.float32, initializer=tf.constant_initializer(0.1)
            )
            out = tf.nn.bias_add(net, biases)
        model_info["inputs"] = inputs
        model_info["outputs"] = [out]
        model_info["in_graph"] = g

    cfg_file = testing.get_model_cfg_path(model_info, "tf")
    input_data = testing.get_op_input(model_name, input_shapes)
    npu_output = testing.get_tvm_output(cfg_file, input_data)
    testing.get_test_result(testing.TFModel(cfg_file), input_data, npu_output, 0.99)
