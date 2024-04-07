# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.parametrize(
    "method_v",
    [
        "Max",
        "Min",
    ],
)
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
        ([[5, 10, 224], [5, 1, 1]], [[5, 10, 224]]),
        ([[5, 10, 224, 100], [10, 224, 1]], [[5, 10, 224, 100]]),
    ],
)
@pytest.mark.parametrize(
    "opset_id",
    [
        8,
    ],
)
def test_max_min(method_v, input_shapes, output_shapes, opset_id):
    if opset_id == 6 and input_shapes[0] != input_shapes[1]:
        pytest.xfail("Opset 6 not support broadcast")

    op_type = method_v
    short_name = f"Eltwise{method_v}"
    dims = []
    for i in input_shapes:
        dims.append(len(i))
    if min(dims) == 0:
        dim_info = "scalar"
    else:
        dim_info = f"{max(dims)}d"
    oos = "OutOfSpec" if max(dims) == 1 else ""
    model_name = aipu_testing.gen_model_name(short_name, dim_info, input_shapes, oos)

    inputs_info = {"shapes": input_shapes}
    outputs_info = {"shapes": output_shapes}

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
