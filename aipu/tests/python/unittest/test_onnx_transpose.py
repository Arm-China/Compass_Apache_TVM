# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
import copy
import pytest
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def get_out_shapes(data, perm):
    assert isinstance(data, list)
    in_shape = copy.deepcopy(data)
    if perm == "default":
        in_shape.reverse()
        return [in_shape]
    assert len(in_shape) == len(perm)
    out_shape = []
    for p in perm:
        out_shape.append(in_shape[p])
    return [out_shape]


@pytest.mark.parametrize(
    "input_shapes, perm",
    [
        ([[2, 3]], [1, 0]),
        ([[1, 2, 3]], [0, 2, 1]),
        ([[1, 2, 3, 4]], [1, 3, 0, 2]),
    ],
)
def test_transpose(input_shapes, perm):
    op_type = "Transpose"
    dim_info = f"{len(input_shapes[0])}d"
    model_name = aipu_testing.gen_model_name(op_type, dim_info, perm)

    output_shapes = get_out_shapes(input_shapes[0], perm)

    extend_attrs = {}
    if perm != "default":
        extend_attrs.update({"perm": perm})

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
