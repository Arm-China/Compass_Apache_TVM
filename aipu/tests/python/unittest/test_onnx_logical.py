# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from onnx import TensorProto
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def logical_test_flow(model_name, input_types, input_shapes, output_shapes, op_type, compass_op_type, opset_id):
    output_types = [TensorProto.BOOL]

    inputs_info = {"shapes": input_shapes, "data_types": input_types}
    outputs_info = {"shapes": output_shapes, "data_types": output_types}

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shapes,
        "inputs": inputs_info,
        "outputs": outputs_info,
        "attributes": {},
        "opset": [opset_id],
    }

    cfg_file = aipu_testing.get_model_cfg_path(model_info, "onnx")
    input_data = aipu_testing.get_op_input(model_name, input_shapes)
    aipu_output = aipu_testing.get_tvm_output(cfg_file, input_data)
    aipu_testing.get_test_result(aipu_testing.ONNXModel(cfg_file), input_data, aipu_output, 0.99)


@pytest.mark.parametrize(
    "opset_id",
    [12],
)
@pytest.mark.parametrize("method_v", ["GreaterOrEqual", "LessOrEqual"])
@pytest.mark.parametrize(
    "input_shapes, output_shapes",
    [
        ([[2, 3, 4, 5], []], [[2, 3, 4, 5]]),
        (
            [
                [2, 3, 4, 5],
                [
                    5,
                ],
            ],
            [[2, 3, 4, 5]],
        ),
        ([[4, 5], [2, 3, 4, 5]], [[2, 3, 4, 5]]),
        ([[1, 4, 5], [2, 3, 1, 1]], [[2, 3, 4, 5]]),
        ([[3, 4, 5], [2, 1, 1, 1]], [[2, 3, 4, 5]]),
        ([[5, 10], [5, 10]], [[5, 10]]),
        ([[5, 2, 3], [5, 2, 3]], [[5, 2, 3]]),
        ([[5, 2, 3, 4], [5, 2, 3, 1]], [[5, 2, 3, 4]]),
    ],
)
def test_logical_greaterorequal_lessorequal(method_v, input_shapes, output_shapes, opset_id):
    op_type = method_v
    compass_op_type = f"Logical{method_v}"
    if len(input_shapes[0]) == len(input_shapes[1]):
        dim_info = f"{len(input_shapes[0])}d"
    else:
        dim_info = "broadcast"
    in_dims = [len(i) for i in input_shapes]
    oos = "OutOfSpec" if min(in_dims) == 0 else ""
    if oos:
        pytest.skip(oos)
    model_name = aipu_testing.gen_model_name(compass_op_type, dim_info, input_shapes, opset_id)

    input_types = [TensorProto.FLOAT, TensorProto.FLOAT]

    logical_test_flow(model_name, input_types, input_shapes, output_shapes, op_type, compass_op_type, opset_id)


@pytest.mark.parametrize(
    "opset_id",
    [
        13,
    ],
)
@pytest.mark.parametrize(
    "method_v",
    [
        "Equal",
        "Greater",
        "Less",
    ],
)
@pytest.mark.parametrize(
    "input_shapes, output_shapes",
    [
        (
            [
                [2, 3, 4, 5],
                [
                    5,
                ],
            ],
            [[2, 3, 4, 5]],
        ),
        ([[4, 5], [2, 3, 4, 5]], [[2, 3, 4, 5]]),
        ([[1, 4, 5], [2, 3, 1, 1]], [[2, 3, 4, 5]]),
        ([[3, 4, 5], [2, 1, 1, 1]], [[2, 3, 4, 5]]),
        ([[5, 10], [5, 10]], [[5, 10]]),
        ([[5, 2, 3], [5, 2, 3]], [[5, 2, 3]]),
        ([[5, 2, 3, 4], [5, 2, 3, 1]], [[5, 2, 3, 4]]),
    ],
)
def test_logical_equal_greater_less(method_v, input_shapes, output_shapes, opset_id):
    op_type = method_v
    compass_op_type = f"Logical{method_v}"
    if len(input_shapes[0]) == len(input_shapes[1]):
        dim_info = f"{len(input_shapes[0])}d"
    else:
        dim_info = "broadcast"
    in_dims = [len(i) for i in input_shapes]
    oos = "OutOfSpec" if min(in_dims) == 0 else ""
    if oos:
        pytest.skip("OutOfSpec")
    model_name = aipu_testing.gen_model_name(compass_op_type, dim_info, input_shapes, opset_id)

    input_types = [TensorProto.FLOAT, TensorProto.FLOAT]

    logical_test_flow(model_name, input_types, input_shapes, output_shapes, op_type, compass_op_type, opset_id)


@pytest.mark.parametrize(
    "opset_id",
    [
        7,
    ],
)
@pytest.mark.parametrize(
    "method_v",
    [
        "And",
        "Or",
        "Xor",
    ],
)
@pytest.mark.parametrize(
    "input_shapes, output_shapes",
    [
        (
            [
                [2, 3, 4, 5],
                [
                    5,
                ],
            ],
            [[2, 3, 4, 5]],
        ),
        ([[4, 5], [2, 3, 4, 5]], [[2, 3, 4, 5]]),
        ([[1, 4, 5], [2, 3, 1, 1]], [[2, 3, 4, 5]]),
        ([[3, 4, 5], [2, 1, 1, 1]], [[2, 3, 4, 5]]),
        ([[5, 10], [5, 10]], [[5, 10]]),
        ([[5, 2, 3], [5, 2, 3]], [[5, 2, 3]]),
        ([[5, 2, 3, 4], [5, 2, 3, 1]], [[5, 2, 3, 4]]),
    ],
)
def test_logical_and_or_xor(method_v, input_shapes, output_shapes, opset_id):
    op_type = method_v
    compass_op_type = f"Logical{method_v}"
    if len(input_shapes[0]) == len(input_shapes[1]):
        dim_info = f"{len(input_shapes[0])}d"
    else:
        dim_info = "broadcast"
    in_dims = [len(i) for i in input_shapes]
    oos = "OutOfSpec" if min(in_dims) == 0 else ""
    if oos:
        pytest.skip("OutOfSpec")
    model_name = aipu_testing.gen_model_name(compass_op_type, dim_info, input_shapes, opset_id)

    input_types = [TensorProto.BOOL, TensorProto.BOOL]

    logical_test_flow(model_name, input_types, input_shapes, output_shapes, op_type, compass_op_type, opset_id)
