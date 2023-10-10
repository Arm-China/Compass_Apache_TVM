# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
import pytest
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.parametrize("opset_id", [1])
@pytest.mark.parametrize(
    "alpha, beta, bias, size",
    [
        (0.4, 0.2, 17.0, 11),
    ],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[10, 20, 30, 40]],
    ],
)
def test_lrn(input_shapes, alpha, beta, bias, size, opset_id):
    op_type = "LRN"
    dim_info = f"{len(input_shapes[0])}d"
    model_name = aipu_testing.gen_model_name(op_type, dim_info, size, opset_id)

    extend_attrs = {"size": size, "alpha": alpha, "beta": beta, "bias": bias}

    inputs_info = {"shapes": input_shapes}
    outputs_info = {"shapes": input_shapes}

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shapes,
        "inputs": inputs_info,
        "outputs": outputs_info,
        "attributes": extend_attrs,
        "opset": [opset_id],
    }

    cfg_file = aipu_testing.get_model_cfg_path(model_info, "onnx")
    input_data = aipu_testing.get_op_input(model_name, input_shapes)
    aipu_output = aipu_testing.get_tvm_output(cfg_file, input_data)
    aipu_testing.get_test_result(aipu_testing.ONNXModel(cfg_file), input_data, aipu_output, 0.99)
