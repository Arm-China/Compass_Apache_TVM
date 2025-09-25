# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from tvm.compass.relax import testing


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
    model_name = testing.gen_model_name(op_type, input_shapes, ksize, strides, padding)

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shapes,
    }

    if not testing.is_model_file_exists(op_type, "tf", model_name):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        c_tf_version = float(".".join(tf.__version__.split(".")[:2]))
        g = tf.Graph()
        with g.as_default():
            inputs = testing.get_input_tensor_of_tf(input_shapes)
            inp = inputs[0]
            if c_tf_version < 1.14:
                out = tf.nn.max_pool(inp, ksize=ksize, strides=strides, padding=padding)
            else:
                out = tf.nn.max_pool2d(inp, ksize=ksize, strides=strides, padding=padding)
        model_info["inputs"] = inputs
        model_info["outputs"] = [out]
        model_info["in_graph"] = g

    cfg_file = testing.get_model_cfg_path(model_info, "tf")
    input_data = testing.get_op_input(model_name, input_shapes)
    npu_output = testing.get_tvm_output(cfg_file, input_data)
    testing.get_test_result(testing.TFModel(cfg_file), input_data, npu_output, 0.99)
