# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.parametrize("opset_id", [11])
@pytest.mark.parametrize("auto_pad", ["default", "SAME_UPPER", "SAME_LOWER", "VALID"])
@pytest.mark.parametrize(
    "group",
    [
        "default",
    ],
)
@pytest.mark.parametrize(
    "dilations",
    [
        "default",
        [2, 4],
    ],
)
@pytest.mark.parametrize("strides", ["default", [3, 4]])
@pytest.mark.parametrize("pads", ["default", [1, 2, 3, 4], [6, 5, 4, 3]])
@pytest.mark.parametrize(
    "kernel_shape",
    [
        [7, 8],
    ],
)
@pytest.mark.parametrize("input_shape", [[2, 6, 51, 52]])
@pytest.mark.parametrize("bias", [True, False])
def test_conv2d(
    input_shape, kernel_shape, strides, pads, dilations, group, auto_pad, bias, opset_id
):
    if auto_pad in ["SAME_UPPER", "SAME_LOWER", "VALID"] and isinstance(pads, list):
        pytest.skip("Can not set pads and auto_pad simultaneously")

    if isinstance(dilations, list):
        if auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
            pytest.skip(
                "Dilation not supported for AutoPadType::SAME_UPPER or AutoPadType::SAME_LOWER"
            )
        if strides != "default":
            pytest.skip(
                "Can not set strides and dilations simultaneously, rule: strides = 1 if dilations"
            )

    op_type = "Conv"
    alias_type = "Conv2D"
    model_name = aipu_testing.gen_model_name(
        alias_type,
        input_shape,
        kernel_shape,
        strides,
        pads,
        dilations,
        group,
        auto_pad,
        bias,
        opset_id,
    )

    extend_attrs = {
        "auto_pad": auto_pad,
        "kernel_shape": kernel_shape,
        "strides": strides,
        "dilations": dilations,
        "pads": pads,
        "group": group,
    }

    spatial_axis_length = len(input_shape[2:])
    if auto_pad == "default":
        del extend_attrs["auto_pad"]
    else:
        del extend_attrs["pads"]
        pads = spatial_axis_length * 2 * [0] if auto_pad == "VALID" else pads
    if dilations == "default":
        del extend_attrs["dilations"]
        dilations = spatial_axis_length * [1]
    if group == "default":
        del extend_attrs["group"]
        group = 1
    if pads == "default":
        if "pads" in extend_attrs:
            del extend_attrs["pads"]
        pads = spatial_axis_length * 2 * [0]
    if strides == "default":
        del extend_attrs["strides"]
        strides = spatial_axis_length * [1]

    shapes_dict = aipu_testing.get_conv_in_out_shapes(
        input_shape, kernel_shape, strides, pads, group, dilations, auto_pad, is_bias=bias
    )

    inputs_info = {
        "shapes": shapes_dict["input_shapes"],
        "default_value": shapes_dict["default_value"],
    }

    outputs_info = {"shapes": shapes_dict["output_shapes"]}

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shape,
        "inputs": inputs_info,
        "outputs": outputs_info,
        "attributes": extend_attrs,
        "opset": [opset_id],
    }

    cfg_file = aipu_testing.get_model_cfg_path(model_info, "onnx")
    input_data = aipu_testing.get_op_input(model_name, input_shape)
    aipu_output = aipu_testing.get_tvm_output(cfg_file, input_data)
    aipu_testing.get_test_result(aipu_testing.ONNXModel(cfg_file), input_data, aipu_output, 0.99)
