# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from tvm.compass.relax import testing


@pytest.mark.parametrize("opset_id", [1])
@pytest.mark.parametrize(
    "op_type",
    [
        "GlobalAveragePool",
        "GlobalMaxPool",
    ],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 20, 30, 40]],
    ],
)
def test_globalpool(input_shapes, op_type, opset_id):
    input_dim = len(input_shapes[0])
    alias_op = "GlobalAvgPool" if op_type == "GlobalAveragePool" else op_type
    dim_info = f"{input_dim}d"
    model_name = testing.gen_model_name(alias_op, dim_info)

    inputs_info = {"shapes": input_shapes}
    outputs_info = {"shapes": [input_shapes[0][:2] + [1] * (input_dim - 2)]}

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shapes,
        "inputs": inputs_info,
        "outputs": outputs_info,
        "attributes": {},
        "opset": [opset_id],
    }

    cfg_file = testing.get_model_cfg_path(model_info, "onnx")
    input_data = testing.get_op_input(model_name, input_shapes)
    npu_output = testing.get_tvm_output(cfg_file, input_data)
    testing.get_test_result(testing.ONNXModel(cfg_file), input_data, npu_output, 0.99)
