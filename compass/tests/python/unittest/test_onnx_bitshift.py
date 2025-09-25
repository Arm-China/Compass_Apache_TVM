# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from tvm.compass.relax import testing


@pytest.mark.parametrize("opset_id", [11])
@pytest.mark.parametrize(
    "input_shapes, output_shapes",
    [
        ([[5, 10], [5, 10]], [[5, 10]]),
        ([[5, 10, 224], [5, 10, 224]], [[5, 10, 224]]),
        ([[5, 10, 224, 100], [5, 10, 224, 100]], [[5, 10, 224, 100]]),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "uint8",
        "uint32",
    ],
)
@pytest.mark.parametrize("direction", ["Left", "Right"])
def test_bitshift(input_shapes, output_shapes, direction, dtype, opset_id):
    op_type = "BitShift"
    dim_info = testing.gen_dim_info(input_shapes[0])
    model_name = testing.gen_model_name(op_type, dim_info, direction, dtype, opset_id)

    onnx_dtype = testing.ONNX_DTYPE_MAPPING[dtype]

    inputs_info = {"shapes": input_shapes, "data_types": [onnx_dtype] * 2}
    outputs_info = {"shapes": output_shapes, "data_types": [onnx_dtype]}

    extend_attrs = {"direction": direction.upper()}

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
    input_data = testing.get_op_input(model_name, input_shapes, {"from_type": dtype})
    npu_output = testing.get_tvm_output(cfg_file, input_data)
    testing.get_test_result(testing.ONNXModel(cfg_file), input_data, npu_output, 0.99)
