# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import copy
import pytest
import tensorflow as tf
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def get_logical_result(inputs: list, method: str):
    if method.lower() == "and":
        return tf.math.logical_and(inputs[0], inputs[1])
    elif method.lower() == "or":
        return tf.math.logical_or(inputs[0], inputs[1])
    elif method.lower() == "not":
        return tf.math.logical_not(inputs[0])
    elif method.lower() == "xor":
        return tf.math.logical_xor(inputs[0], inputs[1])
    elif method.lower() == "equal":
        return tf.math.equal(inputs[0], inputs[1])
    elif method.lower() == "notequal":
        return tf.math.not_equal(inputs[0], inputs[1])
    elif method.lower() == "greater":
        return tf.math.greater(inputs[0], inputs[1])
    elif method.lower() == "greaterequal":
        return tf.math.greater_equal(inputs[0], inputs[1])
    elif method.lower() == "less":
        return tf.math.less(inputs[0], inputs[1])
    elif method.lower() == "lessequal":
        return tf.math.less_equal(inputs[0], inputs[1])
    else:
        raise NotImplementedError(f"{method} Not Implement yet")


@pytest.mark.parametrize(
    "method_v",
    [
        "And",
        "Or",
        "Xor",
        "Not",
        "Equal",
        "Greater",
        "GreaterEqual",
        "NotEqual",
        "Less",
        "LessEqual",
    ],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        ([[2, 3, 4, 5], []]),
        ([[2, 3, 4, 5], [5]]),
        ([[4, 5], [2, 3, 4, 5]]),
        ([[1, 4, 5], [2, 3, 1, 1]]),
        ([[3, 4, 5], [2, 1, 1, 1]]),
        ([[5, 10], [5, 10]]),
        ([[5, 10, 224], [5, 10, 224]]),
        ([[5, 10, 224, 100], [5, 10, 224, 100]]),
    ],
)
def test_logical(input_shapes, method_v):
    input_shapes = copy.deepcopy(input_shapes)

    if method_v in ["And", "Or", "Xor"] and input_shapes == [[2, 3, 4, 5], [5]]:
        pytest.skip("Not Support 1 Dim in Cast OP")

    if method_v == "Not":
        input_shapes.pop()

    op_type = f"Logical{method_v}"
    if len(input_shapes) > 1:
        if len(input_shapes[0]) == len(input_shapes[1]):
            dim_info = f"{len(input_shapes[0])}d"
        else:
            dim_info = "broadcast"
    else:
        dim_info = f"{len(input_shapes[0])}d"
    in_dims = [len(i) for i in input_shapes]
    oos = "OutOfSpec" if min(in_dims) == 0 else ""
    if oos:
        pytest.skip("OutOfSpec")
    model_name = aipu_testing.gen_model_name(op_type, dim_info, input_shapes, "float32")

    g = tf.Graph()
    with g.as_default():
        inputs = aipu_testing.get_input_tensor_of_tf(input_shapes)
        if method_v in ["Equal", "Greater", "GreaterEqual", "NotEqual", "Less", "LessEqual"]:
            _inputs = inputs
        else:
            _inputs = []
            for inp in inputs:
                _inputs.append(tf.cast(inp, tf.bool))
        out = get_logical_result(_inputs, method_v)

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
