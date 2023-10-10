# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
import pytest
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.parametrize("input_shapes, blocksize", [([[1, 2, 12, 18]], 2)])
def test_spacetodepth(input_shapes, blocksize):
    op_type = "SpaceToDepth"
    model_name = aipu_testing.gen_model_name(op_type, f"{len(input_shapes[0])}d", blocksize)

    extend_attrs = {"blocksize": blocksize}

    # output tensor is [N, C * blocksize * blocksize, H/blocksize, W/blocksize]
    n, c, h, w = input_shapes[0]
    output_shapes = [
        [int(n), int(c * blocksize * blocksize), int(h / blocksize), int(w / blocksize)]
    ]

    inputs_info = {"shapes": input_shapes}
    outputs_info = {"shapes": output_shapes}

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shapes,
        "inputs": inputs_info,
        "outputs": outputs_info,
        "attributes": extend_attrs,
        "opset": [],
    }

    cfg_file = aipu_testing.get_model_cfg_path(model_info, "onnx")
    input_data = aipu_testing.get_op_input(model_name, input_shapes)
    aipu_output = aipu_testing.get_tvm_output(cfg_file, input_data)
    aipu_testing.get_test_result(aipu_testing.ONNXModel(cfg_file), input_data, aipu_output, 0.99)
