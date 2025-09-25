# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import copy
import numpy as np
import pytest
from onnx import TensorProto
from tvm.compass.relax import testing


@pytest.mark.parametrize("opset_id", [10])
@pytest.mark.parametrize(
    "batch_axis",
    [
        0,
        1,
    ],
)
@pytest.mark.parametrize(
    "time_axis",
    [
        1,
        0,
    ],
)
@pytest.mark.parametrize(
    "const_input",
    [
        False,
    ],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[5, 6, 7]],
    ],
)
def test_reversesequence(input_shapes, batch_axis, time_axis, const_input, opset_id):
    if testing.skip_case(input_shapes, batch_axis):
        pytest.skip("axis out of input shape")

    if testing.skip_case(input_shapes, time_axis):
        pytest.skip("axis out of input shape")

    extend_paras = {}
    if batch_axis != "default":
        extend_paras["batch_axis"] = batch_axis
    if time_axis != "default":
        extend_paras["time_axis"] = time_axis

    op_type = "ReverseSequence"
    dim_info = f"{len(input_shapes[0])}d"
    if dim_info not in ["2d", "3d"]:
        pytest.skip("OutOfSpec")
    model_name = testing.gen_model_name(op_type, dim_info, batch_axis, time_axis, const_input)

    if batch_axis == "default":
        batch_axis = 1
    if time_axis == "default":
        time_axis = 0

    pos_batch_axis = batch_axis if batch_axis >= 0 else len(input_shapes[0]) + batch_axis

    pos_seq_axis = time_axis if time_axis >= 0 else len(input_shapes[0]) + time_axis

    if pos_batch_axis == pos_seq_axis:
        pytest.skip()

    input_shapes = copy.deepcopy(input_shapes)
    seq_len_shape = [input_shapes[0][pos_batch_axis]]
    seq_len = np.random.randint(1, input_shapes[0][pos_seq_axis] + 1, size=seq_len_shape, dtype=np.int64)
    input_shapes.append(seq_len_shape)

    input_type = [TensorProto.FLOAT, TensorProto.INT64]

    inputs_info = {"shapes": input_shapes, "data_types": input_type}
    if const_input:
        inputs_info["default_value"] = [None, seq_len]

    outputs_info = {"shapes": [input_shapes[0]]}

    other_args = {}
    if not const_input:
        other_args["seq_len"] = seq_len

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shapes,
        "inputs": inputs_info,
        "outputs": outputs_info,
        "attributes": extend_paras,
        "opset": [opset_id],
    }

    cfg_file = testing.get_model_cfg_path(model_info, "onnx")
    input_data = testing.get_op_input(model_name, input_shapes, other_args)
    npu_output = testing.get_tvm_output(cfg_file, input_data)
    testing.get_test_result(testing.ONNXModel(cfg_file), input_data, npu_output, 0.99)
