# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import copy
import random
import pytest
import numpy as np
from onnx import TensorProto
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.parametrize("axis", [1, 2, 3, 4, -1, -2, -3, -4, -5, "default"])
@pytest.mark.parametrize(
    "input_shapes", [[[5, 10]], [[5, 6, 7]], [[5, 6, 7, 8]], [[3, 4, 5, 6, 7]]]
)
@pytest.mark.parametrize("reduction", ["default", "add"])
@pytest.mark.parametrize("opset_id", [13, 16])
def test_scatterelements(input_shapes, axis, reduction, opset_id):
    input_shapes = copy.deepcopy(input_shapes)

    if aipu_testing.skip_case(input_shapes, axis, default_axis=0):
        pytest.skip("axis out of shapes")

    if opset_id < 16 and reduction not in ["default", "none"]:
        pytest.skip()

    op_type = "ScatterElements"
    input_rank = len(input_shapes[0])
    dim_info = f"{input_rank}d"

    # gen indice_shape & updates_shape
    indices_shape = []
    for i in range(input_rank):
        indices_shape.append(random.randint(1, input_shapes[0][i]))

    model_name = aipu_testing.gen_model_name(
        op_type, dim_info, axis, reduction, indices_shape, opset_id
    )

    _axis = axis if axis != "default" else 0

    indice_max_value = input_shapes[0][_axis]  # [-s, s-1]

    indice_value = np.random.randint(
        -indice_max_value, indice_max_value, size=indices_shape, dtype=np.int64
    )

    updates_shape = indices_shape.copy()
    input_shapes.append(indices_shape)
    input_shapes.append(updates_shape)

    extend_paras = {}
    if axis != "default":
        extend_paras["axis"] = axis
    if reduction != "default":
        extend_paras["reduction"] = reduction

    input_type = [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT]
    output_shapes = [input_shapes[0]]

    inputs_info = {"shapes": input_shapes, "data_types": input_type}

    outputs_info = {"shapes": output_shapes}

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
    input_data = aipu_testing.get_op_input(model_name, input_shapes, {"indices_data": indice_value})
    aipu_output = aipu_testing.get_tvm_output(cfg_file, input_data)
    aipu_testing.get_test_result(aipu_testing.ONNXModel(cfg_file), input_data, aipu_output, 0.99)
