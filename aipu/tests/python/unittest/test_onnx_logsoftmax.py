# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.parametrize(
    "opset_id",
    [
        13,
    ],
)
@pytest.mark.parametrize(
    "axis",
    [
        0,
        1,
        2,
        3,
        -1,
        -2,
        -3,
        -4,
    ],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[2, 3]],
        [[2, 3, 4]],
        [[2, 3, 4, 5]],
    ],
)
def test_logsoftmax(input_shapes, axis, opset_id):
    if aipu_testing.skip_case(input_shapes, axis):
        pytest.xfail()

    op_type = "LogSoftmax"
    dim_info = f"{len(input_shapes[0])}d"
    model_name = aipu_testing.gen_model_name(op_type, dim_info, axis, opset_id)

    extend_paras = {}
    if axis != "default":
        extend_paras = {"axis": axis}

    inputs_info = {"shapes": input_shapes}
    outputs_info = {"shapes": input_shapes}

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shapes,
        "inputs": inputs_info,
        "outputs": outputs_info,
        "attributes": extend_paras,
        "opset": [opset_id],
    }

    cfg_file = aipu_testing.get_model_cfg_path(model_info, "onnx")
    input_data = aipu_testing.get_op_input(model_name, input_shapes)
    aipu_output = aipu_testing.get_tvm_output(cfg_file, input_data)
    aipu_testing.get_test_result(aipu_testing.ONNXModel(cfg_file), input_data, aipu_output, 0.99)
