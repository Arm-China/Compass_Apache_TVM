# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
import random
import pytest
import tensorflow as tf
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.parametrize("axis_v", [0, 1, 2, 3, 4, -1, "default"])
@pytest.mark.parametrize("off_value_v, on_value_v", [[3, 7]])
@pytest.mark.parametrize("depth_v", [10, 100])
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[4, 5]],
        [[4, 5, 6]],
        [[4, 5, 6, 7]],
    ],
)
def test_onehot(input_shapes, depth_v, on_value_v, off_value_v, axis_v):
    r = len(input_shapes[0])
    if axis_v != "default" and axis_v not in range(-r - 1, r):
        pytest.xfail("axis must in the range [-r-1, 1]")

    op_type = "OneHot"
    if input_shapes[0]:
        dim_info = f"{len(input_shapes[0])}d"
    else:
        dim_info = "scalar"
    model_name = aipu_testing.gen_model_name(
        op_type, dim_info, depth_v, on_value_v, off_value_v, axis_v, "int32"
    )

    depth_v = random.randint(1, 10) if depth_v == "random" else depth_v
    on_value_v = None if on_value_v == "default" else on_value_v
    off_value_v = None if off_value_v == "default" else off_value_v
    if axis_v == "default":
        axis_v = None
    else:
        axis_v = random.randint(-1, len(input_shapes[0]) - 1) if axis_v == "random" else axis_v

    input_dtypes = [tf.int32]
    g = tf.Graph()
    with g.as_default():
        inputs = aipu_testing.get_input_tensor_of_tf(input_shapes, input_dtypes)
        inp = inputs[0]
        out = tf.one_hot(inp, depth_v, on_value_v, off_value_v, axis_v)

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shapes,
        "inputs": inputs,
        "outputs": [out],
        "in_graph": g,
    }

    cfg_file = aipu_testing.get_model_cfg_path(model_info, "tf")
    input_data = aipu_testing.get_op_input(model_name, input_shapes, op_framework="tf")
    aipu_output = aipu_testing.get_tvm_output(cfg_file, input_data)
    aipu_testing.get_test_result(aipu_testing.TFModel(cfg_file), input_data, aipu_output, 0.99)
