# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
import pytest
import torch
import torchvision
import numpy as np
import tvm
from tvm import relay
from tvm.relay.op.contrib.aipu_compass import deformable_conv2d_v2


def warp_const(inp):
    inp = inp.numpy()
    return relay.Constant(tvm.nd.array(inp))


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 3, 12, 12],
        [2, 16, 24, 24],
        [3, 8, 24, 24],
    ],
)
@pytest.mark.parametrize(
    "kernel_size",
    [2, 3, 4],
)
@pytest.mark.parametrize(
    "output_channels",
    [10, 16, 8],
)
def test_deformable_conv2d_v2(input_shape, kernel_size, output_channels):
    input_tensor = torch.rand(*input_shape)
    IC = input_shape[1]
    weight_tensor = torch.rand(output_channels, IC, kernel_size, kernel_size)
    OH = (input_shape[2] - kernel_size) + 1
    OW = (input_shape[3] - kernel_size) + 1
    offset_tensor = torch.rand(input_shape[0], 2 * kernel_size * kernel_size, OH, OW)
    mask_tensor = torch.rand(input_shape[0], kernel_size * kernel_size, OH, OW)

    torch_out = torchvision.ops.deform_conv2d(
        input_tensor, offset_tensor, weight_tensor, mask=mask_tensor
    ).numpy()

    input_tensor = warp_const(input_tensor)
    weight_tensor = warp_const(weight_tensor)
    offset_tensor = warp_const(offset_tensor)
    mask_tensor = warp_const(mask_tensor)
    expr = deformable_conv2d_v2(input_tensor, offset_tensor, weight_tensor, mask_tensor)
    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)

    desired_layouts = {"contrib.aipu_compass.deformable_conv2d_v2": ["NHWC", "HWIO"]}
    mod = relay.transform.ConvertLayout(desired_layouts)(mod)

    tvm_out = relay.create_executor(device=tvm.cpu(), mod=mod, kind="debug").evaluate()().numpy()
    delta = np.max(np.abs(tvm_out - torch_out))
    assert delta < 10e-4
