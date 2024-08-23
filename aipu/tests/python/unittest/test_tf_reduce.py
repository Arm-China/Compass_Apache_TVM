# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import tensorflow as tf
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def reduce(inp, method, axis, keepdims):
    if method == "All":
        inp = tf.cast(inp, tf.bool)
        return tf.reduce_all(inp, axis=axis, keepdims=keepdims)
    elif method == "Any":
        inp = tf.cast(inp, tf.bool)
        return tf.reduce_any(inp, axis=axis, keepdims=keepdims)
    elif method == "Sum":
        return tf.reduce_sum(inp, axis=axis, keepdims=keepdims)
    elif method == "Mean":
        return tf.reduce_mean(inp, axis=axis, keepdims=keepdims)
    elif method == "Min":
        return tf.reduce_min(inp, axis=axis, keepdims=keepdims)
    elif method == "Max":
        return tf.reduce_max(inp, axis=axis, keepdims=keepdims)
    elif method == "Prod":
        return tf.reduce_prod(inp, axis=axis, keepdims=keepdims)
    elif method == "Variance":
        return tf.math.reduce_variance(inp, axis=axis, keepdims=keepdims)
    elif method == "LogSumExp":
        return tf.math.reduce_logsumexp(inp, axis=axis, keepdims=keepdims)
    else:
        raise NotImplementedError(f"{method} not implemented yet.")


def reduce_test_flow(method_v, input_shapes, axis_v, keepdims_v):
    if aipu_testing.skip_case(input_shapes, axis_v):
        # axis out of input shape
        return

    if method_v == "Prod":
        if axis_v is None or isinstance(axis_v, list):
            # Not Supported in TVM Frontend.
            return

    op_type = f"Reduce{method_v}"
    dim_info = f"{len(input_shapes[0])}d"

    if not keepdims_v and (
        axis_v is None or (isinstance(axis_v, list) and len(axis_v) == len(input_shapes[0])) or dim_info == "1d"
    ):
        # Out of Spec
        return

    model_name = aipu_testing.gen_model_name(op_type, dim_info, axis_v, keepdims_v, "float32")

    g = tf.Graph()
    with g.as_default():
        inputs = aipu_testing.get_input_tensor_of_tf(input_shapes)
        inp = inputs[0]
        out = reduce(inp, method_v, axis_v, keepdims_v)

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


@pytest.mark.parametrize(
    "method_v",
    [
        "Sum",
        "All",
        "Any",
        "Min",
        "Max",
        "Mean",
        "Prod",
        "Variance",
    ],
)
def test_reduce(method_v):
    for axis_v in (0, 1, 2, 3, 4, -1, -2, -3, -4, -5, None, [1, 2], [-2, -3]):
        for keepdims_v in (True, False):
            for input_shapes in ([[2, 3]], [[2, 3, 4]], [[2, 3, 4, 5]]):
                reduce_test_flow(method_v, input_shapes, axis_v, keepdims_v)
