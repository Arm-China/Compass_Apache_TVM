# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def argminmax_test_flow(op_type, input_shapes, axis_v):
    if aipu_testing.skip_case(input_shapes, axis_v):
        # axis out of input shape
        return

    dim_info = f"{len(input_shapes[0])}d"
    model_name = aipu_testing.gen_model_name(op_type, dim_info, axis_v)

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shapes,
    }

    if not aipu_testing.is_model_file_exists(op_type, "tf", model_name):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        g = tf.Graph()
        with g.as_default():
            inputs = aipu_testing.get_input_tensor_of_tf(input_shapes)
            inp = inputs[0]
            func = tf.math.argmax if op_type.lower() == "argmax" else tf.math.argmin
            out = func(inp, axis=axis_v, output_type=tf.dtypes.int32)
        model_info["inputs"] = inputs
        model_info["outputs"] = [out]
        model_info["in_graph"] = g

    cfg_file = aipu_testing.get_model_cfg_path(model_info, "tf")
    input_data = aipu_testing.get_op_input(model_name, input_shapes)
    aipu_output = aipu_testing.get_tvm_output(cfg_file, input_data)
    aipu_testing.get_test_result(aipu_testing.TFModel(cfg_file), input_data, aipu_output, 0.99)


@pytest.mark.parametrize("op_type", ["ArgMax", "ArgMin"])
def test_argminmax(op_type):
    for input_shapes in ([[3, 4]], [[2, 3, 4]], [[2, 3, 4, 5]], [[2, 3, 4, 5, 6]]):
        for axis_v in (0, 1, 2, 3, -1, -2, -3, -4):
            argminmax_test_flow(op_type, input_shapes, axis_v)
