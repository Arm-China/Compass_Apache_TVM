# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
from tvm.compass.relax import testing


@pytest.mark.parametrize(
    "opset_id",
    [
        13,
    ],
)
@pytest.mark.parametrize(
    "fmod",
    [
        1,  # The relay equivalent of np.fmod is relay.mod and np.mod is relay.floor_mod
        "default",  # relay.floor_mod
    ],
)
@pytest.mark.parametrize("dtype", ["uint8", "int8", "uint16", "int16", "uint32", "int32", "int64", "float32"])
@pytest.mark.parametrize(
    "input_shapes, output_shapes",
    [
        ([[2, 3, 4, 5], [2, 3, 4, 5]], [[2, 3, 4, 5]]),
        ([[2, 3, 4, 5], []], [[2, 3, 4, 5]]),
        (
            [
                [2, 3, 4, 5],
                [
                    5,
                ],
            ],
            [[2, 3, 4, 5]],
        ),
        ([[4, 5], [2, 3, 4, 5]], [[2, 3, 4, 5]]),
        ([[1, 4, 5], [2, 3, 1, 1]], [[2, 3, 4, 5]]),
        ([[3, 4, 5], [2, 1, 1, 1]], [[2, 3, 4, 5]]),
    ],
)
def test_mod(input_shapes, output_shapes, fmod, dtype, opset_id):
    # if input dtype is float, fmod must == 1
    if "float" in dtype and fmod != 1:
        pytest.skip()
    if fmod == "default":
        pytest.skip("Not support yet.")

    op_type = "Mod"

    if input_shapes[0] and input_shapes[1]:
        oos = "OutOfSpec" if input_shapes[0][0] != input_shapes[1][0] else ""
    else:
        oos = "OutOfSpec"
    model_name = testing.gen_model_name(op_type, input_shapes, fmod, dtype, oos)

    data_type = testing.ONNX_DTYPE_MAPPING[dtype]

    inputs_info = {"shapes": input_shapes, "data_types": [data_type] * len(input_shapes)}
    outputs_info = {"shapes": output_shapes, "data_types": [data_type]}

    extend_attrs = {}
    if fmod != "default":
        extend_attrs = {"fmod": fmod}

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
