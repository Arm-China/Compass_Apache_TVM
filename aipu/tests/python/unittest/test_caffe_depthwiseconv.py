# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
import os
import random
import pytest
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing

os.environ["GLOG_minloglevel"] = "2"
import caffe  # pylint: disable=wrong-import-position
from caffe import layers as L  # pylint: disable=wrong-import-position


@pytest.mark.parametrize(
    "bias",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "kernel, pad, stride, dilation",  # pad MUST < kernel, stride=1 if dilation
    [
        ([1], [0], [2], 1),
        ([5, 6], [4, 3], [1], 2),
        ([11], [7], [1, 1], 2),
        ([5, 6], [3, 4], [2], 1),
        ([10, 7], [5], [11, 8], 1),
        ([4], [3, 1], [6, 2], 1),
        ([5], [2], [3], 1),
    ],
)
@pytest.mark.parametrize("input_shapes", [[[1, 3, 50, 63]]])
def test_depthwiseconv(input_shapes, kernel, stride, pad, dilation, bias):

    op_type = "DepthwiseConv"
    model_name = aipu_testing.gen_model_name(op_type, kernel, stride, pad, dilation, bias)

    n = caffe.NetSpec()

    n.Placeholder = L.Input(shape=dict(dim=input_shapes[0]), ntop=1)

    group = input_shapes[0][1]

    if "random" in [kernel, stride, pad, dilation]:
        conv_params = aipu_testing.gen_conv_params(
            input_shapes[0], "caffe", group, depthwise=True, dilated=random.choice([True, False])
        )
    else:
        num_output = 15
        conv_params = {
            "dilation": dilation,
            "kernel": kernel,
            "pad": pad,
            "stride": stride,
            "num_output": num_output,
        }

    conv_attr = {
        "dilation": conv_params["dilation"],
        "num_output": conv_params["num_output"],
        "group": group,
        "bias_term": bias,
        "weight_filler": dict(type="gaussian", std=0.01),
        "ntop": 1,
    }

    if bias:
        conv_attr["bias_filler"] = dict(type="constant", value=0.6)

    if len(conv_params["kernel"]) == 1:
        conv_attr["kernel_size"] = conv_params["kernel"][0]
    else:
        conv_attr["kernel_h"] = conv_params["kernel"][0]
        conv_attr["kernel_w"] = conv_params["kernel"][1]

    if len(conv_params["pad"]) == 1:
        conv_attr["pad"] = conv_params["pad"][0]
    else:
        conv_attr["pad_h"] = conv_params["pad"][0]
        conv_attr["pad_w"] = conv_params["pad"][1]

    if len(conv_params["stride"]) == 1:
        conv_attr["stride"] = conv_params["stride"][0]
    else:
        conv_attr["stride_h"] = conv_params["stride"][0]
        conv_attr["stride_w"] = conv_params["stride"][1]

    n.depthwise_conv = L.Convolution(n.Placeholder, **conv_attr)

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shapes,
        "net": n,
        "weights_dict": {},
    }

    cfg_file = aipu_testing.get_model_cfg_path(model_info, "caffe")
    input_data = aipu_testing.get_op_input(model_name, input_shapes)
    aipu_output = aipu_testing.get_tvm_output(cfg_file, input_data)
    aipu_testing.get_test_result(aipu_testing.CaffeModel(cfg_file), input_data, aipu_output, 0.99)
