# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from onnx import TensorProto
from tvm.compass.relax import testing


@pytest.mark.parametrize("opset_id", [6])
@pytest.mark.parametrize(
    "input_shapes, repeats_value",
    [
        ([[2, 3]], [1, 2]),
        ([[2, 3, 4]], [2, 1, 2]),
        ([[2, 3, 4, 5]], [1, 2, 3, 4]),
        ([[2, 3, 4, 5, 6]], [1, 2, 3, 4, 3]),
    ],
)
def test_tile(input_shapes, repeats_value, opset_id):
    op_type = "Tile"
    dim_info = f"{len(input_shapes[0])}d"
    model_name = testing.gen_model_name(op_type, dim_info, opset_id)

    repeats_value_np = np.array(repeats_value, dtype=np.int64)
    repeats_value_ints = [int(x) for x in repeats_value_np]

    # output_dim[i] = input_dim[i] * repeats[i]
    output_shape = []
    for i in range(len(input_shapes[0])):
        output_shape.append(int(input_shapes[0][i] * repeats_value_ints[i]))

    input_shapes.append([len(input_shapes[0])])
    input_types = [TensorProto.FLOAT, TensorProto.INT64]
    output_shapes = [output_shape]

    inputs_info = {
        "shapes": input_shapes,
        "data_types": input_types,
        "default_value": [None, repeats_value_np],
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
