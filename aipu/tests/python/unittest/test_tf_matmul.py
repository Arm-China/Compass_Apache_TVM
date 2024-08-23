# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import tensorflow as tf
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.parametrize(
    "input_shapes, transpose_a_v, transpose_b_v, adjoint_a_v, adjoint_b_v",
    [
        ([[2, 3, 4], [2, 4, 3]], True, True, False, False),
        ([[2, 3, 4], [2, 4, 3]], False, False, True, True),
        ([[2, 3, 4], [2, 4, 3]], True, False, False, True),
        ([[2, 3, 4], [2, 4, 3]], False, True, True, False),
        ([[2, 3, 4], [2, 3, 4]], False, False, True, False),
        ([[2, 3, 4], [2, 3, 4]], False, False, False, True),
        ([[2, 3, 4], [2, 3, 4]], True, False, False, False),
        ([[2, 3, 4], [2, 3, 4]], False, True, False, False),
        ([[2, 3, 4], [2, 4, 3]], False, False, False, False),
        ([[1, 2, 3, 4], [1, 2, 4, 3]], True, True, False, False),
        ([[1, 2, 3, 4], [1, 2, 4, 3]], False, False, True, True),
        ([[1, 2, 3, 4], [1, 2, 4, 3]], True, False, False, True),
        ([[1, 2, 3, 4], [1, 2, 4, 3]], False, True, True, False),
        ([[1, 2, 3, 4], [1, 2, 3, 4]], False, False, True, False),
        ([[1, 2, 3, 4], [1, 2, 3, 4]], False, False, False, True),
        ([[1, 2, 3, 4], [1, 2, 3, 4]], True, False, False, False),
        ([[1, 2, 3, 4], [1, 2, 3, 4]], False, True, False, False),
        ([[1, 2, 3, 4], [1, 2, 4, 3]], False, False, False, False),
        ([[1, 2, 5, 3, 4], [1, 2, 5, 4, 3]], True, True, False, False),
        ([[1, 2, 5, 3, 4], [1, 2, 5, 4, 3]], False, False, True, True),
        ([[1, 2, 5, 3, 4], [1, 2, 5, 4, 3]], True, False, False, True),
        ([[1, 2, 5, 3, 4], [1, 2, 5, 4, 3]], False, True, True, False),
        ([[1, 2, 5, 3, 4], [1, 2, 5, 3, 4]], False, False, True, False),
        ([[1, 2, 5, 3, 4], [1, 2, 5, 3, 4]], False, False, False, True),
        ([[1, 2, 5, 3, 4], [1, 2, 5, 3, 4]], True, False, False, False),
        ([[1, 2, 5, 3, 4], [1, 2, 5, 3, 4]], False, True, False, False),
        ([[1, 2, 5, 3, 4], [1, 2, 5, 4, 3]], False, False, False, False),
    ],
)
def test_matmul(input_shapes, transpose_a_v, transpose_b_v, adjoint_a_v, adjoint_b_v):
    op_type = "MatMul"
    dim_info = f"{len(input_shapes[0])}d"
    if dim_info == "5d":
        pytest.skip("OutOfSpec")
    model_name = aipu_testing.gen_model_name(op_type, dim_info, transpose_a_v, transpose_b_v, adjoint_a_v, adjoint_b_v)

    g = tf.Graph()
    with g.as_default():
        inputs = aipu_testing.get_input_tensor_of_tf(input_shapes)
        inp = inputs[0]
        inp1 = inputs[1]
        out = tf.matmul(
            inp,
            inp1,
            transpose_a=transpose_a_v,
            transpose_b=transpose_b_v,
            adjoint_a=adjoint_a_v,
            adjoint_b=adjoint_b_v,
        )

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
