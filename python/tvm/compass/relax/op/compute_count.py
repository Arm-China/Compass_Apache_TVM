# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Register FComputeCount for some operators."""
import math
from tvm import ir, relax


@ir.register_op_attr("relax.nn.conv2d", "FComputeCount")
@ir.register_op_attr("relax.nn.conv2d_transpose", "FComputeCount")
def _conv2d_mac_count(call: relax.Call):
    attrs = call.attrs
    data_shape = [int(x) for x in call.args[0].struct_info.shape]
    kernel_shape = [int(x) for x in call.args[1].struct_info.shape]
    data_layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    C_ind = data_layout.index("C") if "C" in data_layout else -1  # pylint: disable=invalid-name
    c_ind = data_layout.index("c") if "c" in data_layout else -1
    assert C_ind != -1, "There is no input channel dimension."
    input_channel = data_shape[C_ind]
    input_channel = input_channel * data_shape[c_ind] if c_ind != -1 else input_channel
    kernel_size = [kernel_shape[kernel_layout.index("H")], kernel_shape[kernel_layout.index("W")]]
    output_shape = [int(x) for x in call.struct_info.shape]
    mac_count = math.prod(output_shape) * math.prod(kernel_size)
    assert input_channel % attrs.groups == 0
    mac_count *= input_channel // attrs.groups
    # 1 MAC = 2 FLOPs
    compute_count = mac_count * 2
    return compute_count


@ir.register_op_attr("relax.matmul", "FComputeCount")
def _matmul_mac_count(call: relax.Call):
    inp0_last_dim_value = int(call.args[0].struct_info.shape[-1])
    out_shape = [int(x) for x in call.struct_info.shape]
    return math.prod(out_shape) * inp0_last_dim_value


@ir.register_op_attr("relax.multiply", "FComputeCount")
def _multiply_mac_count(call: relax.Call):
    input_shape = [int(x) for x in call.args[0].struct_info.shape]
    return math.prod(input_shape)
