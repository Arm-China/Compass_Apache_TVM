# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.parametrize(
    "mode",
    [
        "CONSTANT",
        "REFLECT",
        "SYMMETRIC",
    ],
)
def test_pad(mode):
    op_type = "Pad"
    for constant in (0, 2):
        for input_shapes, padding in (
            ([[10, 20]], [[0, 0], [3, 4]]),
            ([[10, 20, 30]], [[0, 0], [3, 4], [5, 6]]),
            ([[10, 20, 30, 40]], [[0, 0], [3, 4], [15, 16], [0, 0]]),
            ([[10, 20, 30, 40, 50]], [[0, 0], [3, 4], [5, 6], [0, 0], [0, 0]]),
        ):
            dim_info = f"{len(input_shapes[0])}d"
            model_name = aipu_testing.gen_model_name(op_type, dim_info, mode, constant)

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
                    padding = tf.constant(padding)
                    out = tf.pad(inp, padding, mode=mode.upper(), constant_values=constant)
                model_info["inputs"] = inputs
                model_info["outputs"] = [out]
                model_info["in_graph"] = g

            cfg_file = aipu_testing.get_model_cfg_path(model_info, "tf")
            input_data = aipu_testing.get_op_input(model_name, input_shapes)
            aipu_output = aipu_testing.get_tvm_output(cfg_file, input_data)
            aipu_testing.get_test_result(aipu_testing.TFModel(cfg_file), input_data, aipu_output, 0.99)
