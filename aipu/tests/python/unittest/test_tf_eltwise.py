# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
import pytest
import tensorflow as tf
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def eltwise(input0, input1, method):
    if method.upper() == "ADD":
        return tf.add(input0, input1)
    elif method.upper() == "SUB":
        return tf.subtract(input0, input1)
    elif method.upper() == "MUL":
        return tf.multiply(input0, input1)
    elif method.upper() == "DIV":
        return tf.math.divide(input0, input1)
    else:
        raise NotImplementedError(f"unsupport method: {method}")


@pytest.mark.parametrize("method", ["Add", "Mul", "Sub", "Div"])
@pytest.mark.parametrize(
    "input_shapes",
    [
        ([[5, 10], [5, 10]]),
        ([[5, 10, 224], [5, 10, 224]]),
        ([[5, 10, 224, 100], [5, 10, 224, 100]]),
    ],
)
def test_eltwise(method, input_shapes):
    op_type = "Eltwise" + method
    dim_info = f"{len(input_shapes[0])}d"
    model_name = aipu_testing.gen_model_name(op_type, dim_info)

    g = tf.Graph()
    with g.as_default():
        inputs = aipu_testing.get_input_tensor_of_tf(input_shapes)
        inp = inputs[0]
        inp2 = inputs[1]
        out = eltwise(inp, inp2, method)

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
