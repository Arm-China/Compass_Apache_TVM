# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
import pytest
import tensorflow as tf
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.parametrize(
    "constant",
    [
        0,
        2,
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "CONSTANT",
        "REFLECT",
        "SYMMETRIC",
    ],
)
@pytest.mark.parametrize(
    "input_shapes, padding",
    [
        ([[10, 20]], [[0, 0], [3, 4]]),
        ([[10, 20, 30]], [[0, 0], [3, 4], [5, 6]]),
        ([[10, 20, 30, 40]], [[0, 0], [3, 4], [15, 16], [0, 0]]),
        ([[10, 20, 30, 40, 50]], [[0, 0], [3, 4], [5, 6], [0, 0], [0, 0]]),
    ],
)
def test_pad(input_shapes, padding, mode, constant):

    op_type = "Pad"
    dim_info = f"{len(input_shapes[0])}d"
    model_name = aipu_testing.gen_model_name(op_type, dim_info, mode, constant)

    g = tf.Graph()
    with g.as_default():
        inputs = aipu_testing.get_input_tensor_of_tf(input_shapes)
        inp = inputs[0]
        padding = tf.constant(padding)
        out = tf.pad(inp, padding, mode=mode.upper(), constant_values=constant)

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
