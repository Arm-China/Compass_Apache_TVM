# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import tensorflow as tf
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.parametrize("input_shapes, blocksize", [([[1, 100, 110, 10]], 2)])
def test_spacetodepth(input_shapes, blocksize):
    op_type = "SpaceToDepth"
    model_name = aipu_testing.gen_model_name(op_type, f"{len(input_shapes[0])}d", blocksize)

    g = tf.Graph()
    with g.as_default():
        inputs = aipu_testing.get_input_tensor_of_tf(input_shapes)
        inp = inputs[0]
        # output tensor is [N, H/blocksize, W/blocksize,  C * blocksize * blocksize]
        out = tf.nn.space_to_depth(inp, blocksize)

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
