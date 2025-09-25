# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import math
import numpy as np
from onnx import TensorProto
from tvm.compass.relax import testing


def infer_output_shape(input_shape, start, end, axes, steps):
    if axes == "default":
        axes = list(range(len(input_shape)))
    if steps == "default":
        steps = [1] * len(input_shape)

    output_shape_dict = {}

    for i, axis in enumerate(axes):
        input_value = input_shape[axis]
        axis = axis if axis >= 0 else len(input_shape) + axis
        start_index = start[i]
        end_index = end[i]
        # If start or end is larger than the n (the number of elements in this dimension), it represents n.
        if start_index < -input_value:
            start_index = -input_value
        if start_index > input_value - 1:
            start_index = input_value - 1
        if end_index < -input_value:
            end_index = -input_value
        if end_index > input_value - 1:
            end_index = input_value - 1
        step = steps[i]
        start_index = start_index if start_index >= 0 else input_value + start_index
        end_index = end_index if end_index >= 0 else input_value + end_index  # exclusive
        if step < 0:
            step = -step
            start_index, end_index = end_index, start_index
        output_shape_dict[axis] = int(math.ceil((end_index - start_index) / step))

    output_shape = []

    for i, shape in enumerate(input_shape):
        if i in output_shape_dict.keys():
            output_shape.append(output_shape_dict[i])
        else:
            output_shape.append(shape)
    return output_shape


@pytest.mark.parametrize(
    "opset_id",
    [
        11,
    ],
)
@pytest.mark.parametrize(
    "input_shapes, starts, ends, axes, steps",
    [
        ([[20, 80]], [4, 5], [19, 78], [0, 1], [1, 1]),
        ([[20, 80, 100]], [70, 40], [80, 60], [-2, -1], [1, 1]),
        ([[20, 80, 100]], [70, 40], [80, 60], [1, 2], [1, 1]),
        ([[10, 20, 100, 100]], [2, 3, 4, 5], [8, 15, 100, 90], [-4, 1, 2, 3], [1, 1, 1, 1]),
        ([[10, 20, 100, 100]], [2, 3, 4, 5], [8, 15, 100, 90], [-4, 1, 2, 3], [3, 1, 1, 5]),
        (
            [[10, 20, 100, 100, 100]],
            [2, 3, 4, 5, 6],
            [8, 15, 100, 90, 50],
            [0, 1, 2, 3, 4],
            [1, 1, 1, 1, 1],
        ),
    ],
)
def test_slice(input_shapes, starts, ends, axes, steps, opset_id):
    output_shapes = [infer_output_shape(input_shapes[0], starts, ends, axes, steps)]

    op_type = "Slice"
    dim_info = f"{len(input_shapes[0])}d"
    model_name = testing.gen_model_name(op_type, dim_info, starts, ends, axes, steps, output_shapes, opset_id)

    input_types = [TensorProto.FLOAT]
    inp_data_list = [None]
    for d in starts, ends, axes, steps:
        if isinstance(d, list):
            input_shapes.append([len(d)])
            input_types.append(TensorProto.INT32)
            inp_data_list.append(np.array(d, dtype=np.int32))

    inputs_info = {
        "shapes": input_shapes,
        "data_types": input_types,
        "default_value": inp_data_list,
    }

    outputs_info = {"shapes": output_shapes}

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shapes[0],
        "inputs": inputs_info,
        "outputs": outputs_info,
        "attributes": {},
        "opset": [opset_id],
    }

    cfg_file = testing.get_model_cfg_path(model_info, "onnx")
    input_data = testing.get_op_input(model_name, input_shapes[0])
    npu_output = testing.get_tvm_output(cfg_file, input_data)
    testing.get_test_result(testing.ONNXModel(cfg_file), input_data, npu_output, 0.99)
