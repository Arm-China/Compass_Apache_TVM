# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import random
from tvm.compass.relax import testing


def infer_output_shape(input_shape, split, axis):
    assert isinstance(input_shape, list), "Input shape should be list type"
    if axis == "default":
        axis = 0
    assert abs(axis) <= len(input_shape), "Axis should <= length of input shape"
    input_value = int(input_shape[axis])  # value to be splited
    if split == "default":  # split to equal sizeed parts
        if input_value in [0, 1]:
            split = [input_value]
        else:
            if input_value % 2 and input_value % 3:  # set split value specifically due to the input_value is odd number
                sub_split = random.randint(0, input_value)
                split = [int(input_value - sub_split), sub_split]
            else:
                split = [2] * int(input_value / 2) if input_value % 3 else [3] * int(input_value / 3)
    assert sum(split) == input_value, "Sum of the split values must be equal to the dim value at axis specified"

    output_shapes = []
    sub_output_shape = input_shape.copy()
    for s in split:
        sub_output_shape[axis] = s
        output_shapes.append(sub_output_shape)
    return output_shapes


@pytest.mark.parametrize(
    "opset_id",
    [
        11,
    ],
)
@pytest.mark.parametrize(
    "axis",
    [
        0,
        1,
        2,
        3,
    ],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[10, 20]],
        [[10, 20, 30]],
        [[10, 20, 30, 40]],
        [[10, 20, 30, 40, 50]],
    ],
)
def test_split(input_shapes, axis, opset_id):
    if testing.skip_case(input_shapes, axis, default_axis=0):
        pytest.skip("axis out of shapes")

    op_type = "Split"
    dim_info = f"{len(input_shapes[0])}d"
    model_name = testing.gen_model_name(op_type, dim_info, axis, opset_id)

    split = [2, int(input_shapes[0][axis] - 2)]
    extend_attrs = {
        "axis": axis,
        "split": split,
    }

    inputs_info = {"shapes": input_shapes}
    output_shapes = infer_output_shape(input_shapes[0], split, axis)
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

    cfg_file = testing.get_model_cfg_path(model_info, "onnx")
    input_data = testing.get_op_input(model_name, input_shapes)
    npu_output = testing.get_tvm_output(cfg_file, input_data)
    testing.get_test_result(testing.ONNXModel(cfg_file), input_data, npu_output, 0.99)
