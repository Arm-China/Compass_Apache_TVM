# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from tvm.compass.relax import testing


@pytest.mark.parametrize(
    "method",
    [
        "Add",
        "Mul",
        "Sub",
    ],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        ([[5, 10], [5, 10]]),
        ([[5, 10, 224], [5, 10, 224]]),
        ([[5, 10, 224, 100], [5, 10, 224, 100]]),
    ],
)
def test_eltwiserelu(method, input_shapes):
    op_type = "EltwiseRelu"
    dim_info = f"{len(input_shapes[0])}d"
    model_name = testing.gen_model_name(op_type, dim_info, method)

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
            inp2 = inputs[1]
            if method.upper() == "ADD":
                out_eltwise = tf.add(inp, inp2)
            elif method.upper() == "SUB":
                out_eltwise = tf.subtract(inp, inp2)
            elif method.upper() == "MUL":
                out_eltwise = tf.multiply(inp, inp2)
            out = tf.nn.relu(out_eltwise)
        model_info["inputs"] = inputs
        model_info["outputs"] = [out]
        model_info["in_graph"] = g

    cfg_file = testing.get_model_cfg_path(model_info, "tf")
    input_data = testing.get_op_input(model_name, input_shapes)
    npu_output = testing.get_tvm_output(cfg_file, input_data)
    testing.get_test_result(testing.TFModel(cfg_file), input_data, npu_output, 0.99)
