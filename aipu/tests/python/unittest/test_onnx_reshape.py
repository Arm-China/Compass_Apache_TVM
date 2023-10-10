# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from onnx import TensorProto
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.parametrize("opset_id", [5])
@pytest.mark.parametrize(
    "input_shape, shape",
    [
        ([120], [2, 3, 4, 5]),
        ([1, 300], [1, 10, 10, -1]),
        ([1, 100, 3], [1, 10, 10, 3]),
        ([1, 1, 1, 3], [1, 1, 3]),
        ([1, 10, 20, 30, 3], [1, 0, 20, 90]),
    ],
)
def test_reshape(input_shape, shape, opset_id):
    op_type = "Reshape"
    dim_info = f"{len(input_shape)}d"
    model_name = aipu_testing.gen_model_name(op_type, dim_info, opset_id)

    input_shapes = [input_shape, [len(shape)]]
    input_types = [TensorProto.FLOAT, TensorProto.INT64]
    shape_data = np.array(shape, dtype=np.int64)
    inputs_info = {
        "shapes": input_shapes,
        "data_types": input_types,
        "default_value": [None, shape_data],
    }

    if dim_info == "5d":
        # numpy.reshape not support zero in newshape.
        output_shapes = [[1, 10, 20, 90]]
    else:
        input_shape_np = np.empty(input_shape, dtype=int)
        output_shape_np = np.reshape(input_shape_np, tuple(shape))
        output_shapes = [list(output_shape_np.shape)]
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

    cfg_file = aipu_testing.get_model_cfg_path(model_info, "onnx")
    input_data = aipu_testing.get_op_input(model_name, input_shapes[0])
    aipu_output = aipu_testing.get_tvm_output(cfg_file, input_data)
    aipu_testing.get_test_result(aipu_testing.ONNXModel(cfg_file), input_data, aipu_output, 0.99)
