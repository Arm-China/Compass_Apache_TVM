# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from tvm.compass.relax import testing


@pytest.mark.parametrize("opset_id", [11])
@pytest.mark.parametrize(
    "auto_pad_v",
    [
        "default",
        "VALID",
    ],
)
@pytest.mark.parametrize(
    "group_v",
    [
        "default",
    ],
)
@pytest.mark.parametrize(
    "output_shape_v",
    [
        "default",
    ],
)
@pytest.mark.parametrize(
    "output_padding_v",
    [
        "default",
    ],
)
@pytest.mark.parametrize("dilations_v", ["default", [2, 2], [3, 4]])
@pytest.mark.parametrize(
    "strides_v",
    [
        "default",
        [3, 4],
    ],
)
@pytest.mark.parametrize("pads_v", ["default", [1, 2, 3, 4], [6, 5, 4, 3]])
@pytest.mark.parametrize("kernel_shape_v", [[7, 8]])
@pytest.mark.parametrize("input_shape", [[2, 6, 57, 58]])
@pytest.mark.parametrize("bias", [True, False])
def test_convtranspose(
    input_shape,
    kernel_shape_v,
    strides_v,
    pads_v,
    dilations_v,
    group_v,
    output_padding_v,
    output_shape_v,
    auto_pad_v,
    bias,
    opset_id,
):
    if auto_pad_v in ["SAME_UPPER", "SAME_LOWER", "VALID"] and isinstance(pads_v, list):
        pytest.skip("Can not set pads and auto_pad simultaneously")

    if strides_v != "default" and isinstance(dilations_v, list):
        pytest.skip("strides = 1 if dilations")

    if auto_pad_v in ["SAME_UPPER", "SAME_LOWER"] and isinstance(dilations_v, list):
        pytest.skip("Dilation NOT support in AutoPad: SAME")

    op_type = "ConvTranspose"
    alias_type = "ConvTranspose2D"
    model_name = testing.gen_model_name(
        alias_type,
        input_shape,
        kernel_shape_v,
        strides_v,
        pads_v,
        dilations_v,
        group_v,
        output_padding_v,
        output_shape_v,
        auto_pad_v,
        bias,
        opset_id,
    )

    extend_paras = {
        "auto_pad": auto_pad_v,
        "dilations": dilations_v,
        "group": group_v,
        "kernel_shape": kernel_shape_v,
        "output_padding": output_padding_v,
        "output_shape": output_shape_v,
        "pads": pads_v,
        "strides": strides_v,
    }

    spatial_axis_length = len(input_shape[2:])
    if auto_pad_v == "default":
        del extend_paras["auto_pad"]
    else:
        del extend_paras["pads"]
        pads_v = spatial_axis_length * 2 * [0] if auto_pad_v == "VALID" else pads_v
    if dilations_v == "default":
        del extend_paras["dilations"]
        dilations_v = spatial_axis_length * [1]
    if group_v == "default":
        del extend_paras["group"]
        group_v = 1
    if output_padding_v == "default":
        del extend_paras["output_padding"]
        output_padding_v = spatial_axis_length * [0]
    else:
        if len(output_padding_v) != spatial_axis_length:
            output_padding_v = spatial_axis_length * [0]
    if output_shape_v == "default":
        del extend_paras["output_shape"]
    if pads_v == "default":
        if "pads" in extend_paras.keys():
            del extend_paras["pads"]
        pads_v = spatial_axis_length * 2 * [0]
    if strides_v == "default":
        del extend_paras["strides"]
        strides_v = spatial_axis_length * [1]

    shapes_dict = testing.get_conv_in_out_shapes(
        input_shape,
        kernel_shape_v,
        strides_v,
        pads_v,
        group_v,
        dilations_v,
        auto_pad_v,
        output_padding=output_padding_v,
        output_shape=output_shape_v,
        is_bias=bias,
        is_trans=True,
    )

    inputs_info = {
        "shapes": shapes_dict["input_shapes"],
        "default_value": shapes_dict["default_value"],
    }

    outputs_info = {"shapes": shapes_dict["output_shapes"]}

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": shapes_dict["input_shapes"][0],
        "inputs": inputs_info,
        "outputs": outputs_info,
        "attributes": extend_paras,
        "opset": [opset_id],
    }

    cfg_file = testing.get_model_cfg_path(model_info, "onnx")
    input_data = testing.get_op_input(model_name, shapes_dict["input_shapes"][0])
    npu_output = testing.get_tvm_output(cfg_file, input_data)
    testing.get_test_result(testing.ONNXModel(cfg_file), input_data, npu_output, 0.99)
