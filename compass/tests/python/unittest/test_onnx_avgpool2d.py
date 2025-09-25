# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import math
import pytest
from tvm.compass.relax import testing


@pytest.mark.parametrize(
    "opset_id",
    [
        11,
    ],
)
@pytest.mark.parametrize("ceil_mode", [1, "default"])
@pytest.mark.parametrize("count_include_pad", [1, "default"])
@pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "SAME_LOWER", "VALID", "default"])
@pytest.mark.parametrize("input_shapes", [([[10, 15, 500, 224]])])
@pytest.mark.parametrize(
    "kernel_shape",
    [
        [7, 8],
    ],
)
@pytest.mark.parametrize(
    "pads",
    [
        "default",
        [1, 2, 3, 4],
        [4, 3, 2, 1],
    ],
)
@pytest.mark.parametrize(
    "strides",
    [
        "default",
        [2, 3],
        [3, 1],
    ],
)
def test_avgpool2d(input_shapes, kernel_shape, pads, strides, auto_pad, count_include_pad, ceil_mode, opset_id):
    if auto_pad not in ["NOTSET", "default"] and isinstance(pads, list):
        pytest.skip("pads cannot be used simultaneously with auto_pad attribute")

    if auto_pad == "VALID" and ceil_mode == 1:
        pytest.skip("OnnxRT implementation not match with onnx op SPEC")
        # https://github.com/microsoft/onnxruntime/issues/10083

    if auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
        # cal actual pad
        _strides = [1, 1] if strides == "default" else strides.copy()
        out_h = math.ceil(input_shapes[0][2] / _strides[0])
        out_w = math.ceil(input_shapes[0][3] / _strides[1])
        pad_h = (out_h - 1) * _strides[0] + kernel_shape[0] - input_shapes[0][2]
        pad_w = (out_w - 1) * _strides[1] + kernel_shape[1] - input_shapes[0][3]
        if (pad_h < 0 and abs(pad_h) % 2 == 1) or (pad_w < 0 and abs(pad_w) % 2 == 1):
            pytest.skip("ORT Bug")
            # https://github.com/microsoft/onnxruntime/issues/10086

    op_type = "AveragePool"
    alias_type = "AvgPool2D"
    model_name = testing.gen_model_name(
        alias_type, kernel_shape, pads, strides, auto_pad, count_include_pad, ceil_mode, opset_id
    )

    extend_attrs = {"kernel_shape": kernel_shape}
    if auto_pad != "default":
        extend_attrs.update({"auto_pad": auto_pad})
    if ceil_mode != "default":
        extend_attrs.update({"ceil_mode": ceil_mode})
    if count_include_pad != "default":
        extend_attrs.update({"count_include_pad": count_include_pad})
    if pads != "default":
        extend_attrs.update({"pads": pads})
    if strides != "default":
        extend_attrs.update({"strides": strides})

    output_shapes = testing.get_pool_out_shapes(auto_pad, input_shapes, kernel_shape, strides, ceil_mode, pads, "avg")

    if ceil_mode == 1:
        _strides = [1, 1] if strides == "default" else strides.copy()
        dilation = [1, 1]
        _pad = [0, 0, 0, 0] if pads == "default" else pads.copy()
        extra_ph = (
            (output_shapes[0][2] - 1) * _strides[0]
            + dilation[0] * (kernel_shape[0] - 1)
            + 1
            - input_shapes[0][2]
            - _pad[0]
            - _pad[2]
        )
        extra_pw = (
            (output_shapes[0][3] - 1) * _strides[1]
            + dilation[1] * (kernel_shape[1] - 1)
            + 1
            - input_shapes[0][3]
            - _pad[1]
            - _pad[3]
        )
        if (extra_ph + _pad[2] >= kernel_shape[0]) or (extra_pw + _pad[3] >= kernel_shape[1]):
            pytest.skip("Out Of Spec")

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

    cfg_file = testing.get_model_cfg_path(model_info, "onnx")
    input_data = testing.get_op_input(model_name, input_shapes)
    npu_output = testing.get_tvm_output(cfg_file, input_data)
    testing.get_test_result(testing.ONNXModel(cfg_file), input_data, npu_output, 0.99)
