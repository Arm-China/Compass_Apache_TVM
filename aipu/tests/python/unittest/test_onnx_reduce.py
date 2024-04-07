# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def get_output_shapes(input_shape, axes, keepdims=True):
    axes = tuple(range(len(input_shape))) if axes == "default" else tuple(axes)
    output_shape = input_shape.copy()

    if keepdims:
        for a in axes:
            output_shape[a] = 1
        output_shapes = [output_shape]
    else:
        output_shapes = [[j for i, j in enumerate(output_shape) if i not in axes]]

    return output_shapes


@pytest.mark.parametrize("opset_id", [11])
@pytest.mark.parametrize(
    "axes",
    [
        [0],
        [1],
        [2],
        [3],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
        [0, 1, 2, 3],
    ],
)
@pytest.mark.parametrize(
    "method",
    [
        "Mean",
    ],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[10, 20, 30, 40]],
    ],
)
def test_reduce(input_shapes, method, axes, opset_id):
    op_type = "Reduce" + str(method)
    model_name = aipu_testing.gen_model_name(op_type, input_shapes, axes, opset_id)

    extend_attrs = {
        "axes": axes,
        "keepdims": 1,  # keep reduced dimension
    }

    output_shapes = get_output_shapes(input_shapes[0], axes, True)

    inputs_info = {"shapes": input_shapes}
    outputs_info = {"shapes": output_shapes}

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
