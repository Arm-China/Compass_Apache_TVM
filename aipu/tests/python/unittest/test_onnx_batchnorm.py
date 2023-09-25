# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
import numpy as np
import pytest
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.parametrize("opset_id", [9])
@pytest.mark.parametrize(
    "input_shapes,eps",
    [
        ([[3, 4, 5], [4], [4], [4], [4]], 0.8),
        ([[5, 3, 65, 56], [3], [3], [3], [3]], 0.56),
        ([[4, 5, 244, 100], [5], [5], [5], [5]], "default"),
    ],
)
def test_batchnorm(input_shapes, eps, opset_id):
    # Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
    op_type = "BatchNormalization"
    alias_type = "BatchNorm"
    model_name = aipu_testing.gen_model_name(alias_type, input_shapes[0], eps, opset_id)

    extend_attrs = {"epsilon": eps} if eps != "default" else {}

    # we only support const gamma/beta
    scale = np.random.random(tuple(input_shapes[1])).astype(np.float32)
    bias = np.random.random(tuple(input_shapes[2])).astype(np.float32)
    inp_mean = np.random.random(tuple(input_shapes[3])).astype(np.float32)
    inp_var = np.random.random(tuple(input_shapes[4])).astype(np.float32)

    inputs_info = {"shapes": input_shapes, "default_value": [None, scale, bias, inp_mean, inp_var]}
    outputs_info = {"shapes": [input_shapes[0]]}

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": [input_shapes[0]],
        "inputs": inputs_info,
        "outputs": outputs_info,
        "attributes": extend_attrs,
        "opset": [opset_id],
    }

    cfg_file = aipu_testing.get_model_cfg_path(model_info, "onnx")
    input_data = aipu_testing.get_op_input(model_name, [input_shapes[0]])
    aipu_output = aipu_testing.get_tvm_output(cfg_file, input_data)
    aipu_testing.get_test_result(aipu_testing.ONNXModel(cfg_file), input_data, aipu_output, 0.99)
