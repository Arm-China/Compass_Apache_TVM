# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
import pytest
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.parametrize(
    "opset_id",
    [
        11,
    ],
)
@pytest.mark.parametrize(
    "mode_v",
    [
        "DCR",
        "default",
    ],
)
@pytest.mark.parametrize(
    "blocksize_v",
    [2, 3],
)
@pytest.mark.parametrize("input_shapes", [[[2, 72, 4, 5]]])
def test_depthtospace(input_shapes, blocksize_v, mode_v, opset_id):
    if opset_id == 1 and mode_v != "default":
        pytest.xfail("Mode only supported in OPSet > 1")

    op_type = "DepthToSpace"
    model_name = aipu_testing.gen_model_name(op_type, blocksize_v, mode_v, opset_id)

    extend_paras = {"blocksize": blocksize_v}
    if mode_v != "default":
        extend_paras.update({"mode": mode_v})

    inputs_info = {"shapes": input_shapes}

    n, c, h, w = input_shapes[0]
    output_shapes = [n, c / (blocksize_v * 2), h * blocksize_v, w * blocksize_v]
    output_shapes = list(map(int, output_shapes))
    outputs_info = {"shapes": [output_shapes]}

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
