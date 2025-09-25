# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from tvm.compass.relax import testing


@pytest.mark.parametrize(
    "method",
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
def test_logical(method):
    for input_shapes in (
        [[2, 3, 4, 5], []],
        [[2, 3, 4, 5], [5]],
        [[4, 5], [2, 3, 4, 5]],
        [[1, 4, 5], [2, 3, 1, 1]],
        [[3, 4, 5], [2, 1, 1, 1]],
        [[5, 10], [5, 10]],
        [[5, 10, 224], [5, 10, 224]],
        [[5, 10, 224, 100], [5, 10, 224, 100]],
    ):
        if method in ["And", "Or", "Xor"] and input_shapes == [[2, 3, 4, 5], [5]]:
            # Not Support 1 Dim in Cast OP
            continue

        if method == "Not":
            input_shapes.pop()

        op_type = f"Logical{method}"
        if len(input_shapes) > 1:
            if len(input_shapes[0]) == len(input_shapes[1]):
                dim_info = f"{len(input_shapes[0])}d"
            else:
                dim_info = "broadcast"
        else:
            dim_info = f"{len(input_shapes[0])}d"
        in_dims = [len(i) for i in input_shapes]
        if min(in_dims) == 0:  # Out of Spec
            continue
        model_name = testing.gen_model_name(op_type, dim_info, input_shapes, "float32")

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
                if method in ["Equal", "Greater", "GreaterEqual", "NotEqual", "Less", "LessEqual"]:
                    _inputs = inputs
                else:
                    _inputs = []
                    for inp in inputs:
                        _inputs.append(tf.cast(inp, tf.bool))
                if method.lower() == "and":
                    out = tf.math.logical_and(_inputs[0], _inputs[1])
                elif method.lower() == "or":
                    out = tf.math.logical_or(_inputs[0], _inputs[1])
                elif method.lower() == "not":
                    out = tf.math.logical_not(_inputs[0])
                elif method.lower() == "xor":
                    out = tf.math.logical_xor(_inputs[0], _inputs[1])
                elif method.lower() == "equal":
                    out = tf.math.equal(_inputs[0], _inputs[1])
                elif method.lower() == "notequal":
                    out = tf.math.not_equal(_inputs[0], _inputs[1])
                elif method.lower() == "greater":
                    out = tf.math.greater(_inputs[0], _inputs[1])
                elif method.lower() == "greaterequal":
                    out = tf.math.greater_equal(_inputs[0], _inputs[1])
                elif method.lower() == "less":
                    out = tf.math.less(_inputs[0], _inputs[1])
                elif method.lower() == "lessequal":
                    out = tf.math.less_equal(_inputs[0], _inputs[1])
            model_info["inputs"] = inputs
            model_info["outputs"] = [out]
            model_info["in_graph"] = g

        cfg_file = testing.get_model_cfg_path(model_info, "tf")
        input_data = testing.get_op_input(model_name, input_shapes, op_framework="tf")
        npu_output = testing.get_tvm_output(cfg_file, input_data)
        testing.get_test_result(testing.TFModel(cfg_file), input_data, npu_output, 0.99)
