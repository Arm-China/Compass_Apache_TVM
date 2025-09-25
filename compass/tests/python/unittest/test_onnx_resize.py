# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import copy
import numpy as np
import pytest
from tvm.compass.relax import testing


@pytest.mark.parametrize(
    "opset_id",
    [
        11,
    ],
)
@pytest.mark.parametrize(
    "coordinate_transformation_mode",
    [
        "align_corners",
        "asymmetric",
        "half_pixel",
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "linear",
        "nearest",
    ],
)
@pytest.mark.parametrize(
    "scales",
    [
        ([1.0, 1.0, 1.5, 1.8]),
    ],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[2, 3, 10, 20]],
    ],
)
def test_resize(input_shapes, mode, coordinate_transformation_mode, scales, opset_id):
    op_type = "Resize"
    model_name = testing.gen_model_name(op_type, mode, coordinate_transformation_mode, scales, opset_id)

    extend_attrs = {
        "mode": mode,
        "coordinate_transformation_mode": coordinate_transformation_mode,
    }

    default_values = [None]
    input_shapes = copy.deepcopy(input_shapes)

    # inputs: roi
    default_values.append(np.array([], dtype=float))
    input_shapes.append([0])

    # inputs: scales
    if scales:
        inp_scales = np.array(scales, dtype=np.float32)
        default_values.append(inp_scales)
        scales_shape = [int(len(scales))] if len(scales) > 0 else "_"
        input_shapes.append(scales_shape)
    else:
        default_values.append(np.array([], dtype=float))
        input_shapes.append([0])

    inputs_info = {"shapes": input_shapes, "default_value": default_values}

    output_shapes = [[int(x * y) for x, y in zip(input_shapes[0], scales)]]
    outputs_info = {"shapes": output_shapes}

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shapes[0],
        "inputs": inputs_info,
        "outputs": outputs_info,
        "attributes": extend_attrs,
        "opset": [opset_id],
    }

    cfg_file = testing.get_model_cfg_path(model_info, "onnx")
    input_data = testing.get_op_input(model_name, input_shapes[0])
    npu_output = testing.get_tvm_output(cfg_file, input_data)
    testing.get_test_result(testing.ONNXModel(cfg_file), input_data, npu_output, 0.99)
