# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
import pytest
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.parametrize(
    "opset_id",
    [
        13,
    ],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 20]],
        [[1, 20, 30]],
        [[1, 20, 30, 40]],
        [[1, 20, 30, 40, 50]],
    ],
)
def test_sigmoid(input_shapes, opset_id):
    op_type = "Sigmoid"
    model_name = aipu_testing.gen_model_name(
        op_type, aipu_testing.gen_dim_info(input_shapes), opset_id
    )

    inputs_info = {"shapes": input_shapes}
    outputs_info = {"shapes": input_shapes}

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
