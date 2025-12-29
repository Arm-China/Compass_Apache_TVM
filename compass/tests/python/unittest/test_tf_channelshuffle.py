# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
from tvm.compass.relax import testing


@pytest.mark.parametrize("input_shapes", [[[2, 3, 6, 12]]])
@pytest.mark.parametrize(
    "splits",
    [
        1,
        4,
    ],
)
def test_channelshuffle(input_shapes, splits):
    op_type = "ChannelShuffle"
    model_name = testing.gen_model_name(op_type, input_shapes, splits)

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
            n, h, w, c = input_shapes[0]
            groups = 3
            assert splits <= c // groups, "split must be 1 or input_shape[-1]/group"
            reshape_0 = tf.reshape(inp, (n, -1, groups, c // groups))
            transpose = tf.transpose(reshape_0, (0, 1, 3, 2))
            output = tf.reshape(transpose, (n, h, w, c))
            split_outputs = tf.split(output, num_or_size_splits=splits, axis=-1)
        model_info["inputs"] = inputs
        model_info["outputs"] = split_outputs
        model_info["in_graph"] = g

    cfg_file = testing.get_model_cfg_path(model_info, "tf")
    input_data = testing.get_op_input(model_name, input_shapes)
    npu_output = testing.get_tvm_output(cfg_file, input_data)
    testing.get_test_result(testing.TFModel(cfg_file), input_data, npu_output, 0.99)
