# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import random
from tvm.compass.relax import testing


def onehot_test_flow(input_shapes, depth_v, on_value_v, off_value_v, axis_v):
    r = len(input_shapes[0])
    if axis_v != "default" and axis_v not in range(-r - 1, r):
        # axis must in the range [-r-1, 1]
        return

    op_type = "OneHot"
    if input_shapes[0]:
        dim_info = f"{len(input_shapes[0])}d"
    else:
        dim_info = "scalar"
    model_name = testing.gen_model_name(op_type, dim_info, depth_v, on_value_v, off_value_v, axis_v, "int32")

    depth_v = random.randint(1, 10) if depth_v == "random" else depth_v
    on_value_v = None if on_value_v == "default" else on_value_v
    off_value_v = None if off_value_v == "default" else off_value_v
    if axis_v == "default":
        axis_v = None
    else:
        axis_v = random.randint(-1, len(input_shapes[0]) - 1) if axis_v == "random" else axis_v

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shapes,
    }

    if not testing.is_model_file_exists(op_type, "tf", model_name):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        input_dtypes = [tf.int32]
        g = tf.Graph()
        with g.as_default():
            inputs = testing.get_input_tensor_of_tf(input_shapes, input_dtypes)
            inp = inputs[0]
            out = tf.one_hot(inp, depth_v, on_value_v, off_value_v, axis_v)
        model_info["inputs"] = inputs
        model_info["outputs"] = [out]
        model_info["in_graph"] = g

    cfg_file = testing.get_model_cfg_path(model_info, "tf")
    input_data = testing.get_op_input(model_name, input_shapes, op_framework="tf")
    npu_output = testing.get_tvm_output(cfg_file, input_data)
    testing.get_test_result(testing.TFModel(cfg_file), input_data, npu_output, 0.99)


def test_onehot():
    for axis_v in [0, 1, 2, 3, 4, -1, "default"]:
        for off_value_v, on_value_v in [[3, 7]]:
            for depth_v in [10, 100]:
                for input_shapes in ([[4, 5]], [[4, 5, 6]], [[4, 5, 6, 7]]):
                    onehot_test_flow(input_shapes, depth_v, on_value_v, off_value_v, axis_v)
