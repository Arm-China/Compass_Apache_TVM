# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
import pytest
from onnx import TensorProto
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


@pytest.mark.parametrize(
    "from_type, to_type",
    [
        ("FLOAT", "INT8"),
        ("FLOAT", "UINT8"),
        ("FLOAT", "INT16"),
        ("FLOAT", "UINT16"),
        ("FLOAT", "INT32"),
        ("FLOAT", "FLOAT16"),
    ],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[2, 3]],
        [[2, 3, 4]],
        [[2, 3, 4, 5]],
    ],
)
def test_cast(input_shapes, from_type, to_type):
    op_type = "Cast"
    dim_info = f"{len(input_shapes[0])}d"
    model_name = aipu_testing.gen_model_name(op_type, dim_info, from_type, to_type)

    extend_attrs = {"to": getattr(TensorProto, to_type)}

    input_types = [aipu_testing.ONNX_DTYPE_MAPPING[from_type.lower()]]
    output_types = [aipu_testing.ONNX_DTYPE_MAPPING[to_type.lower()]]
    inputs_info = {"shapes": input_shapes, "data_types": input_types}
    outputs_info = {"shapes": input_shapes, "data_types": output_types}

    model_info = {
        "model_name": model_name,
        "op_type": op_type,
        "input_shapes": input_shapes,
        "inputs": inputs_info,
        "outputs": outputs_info,
        "attributes": extend_attrs,
        "opset": [],
    }

    cfg_file = aipu_testing.get_model_cfg_path(model_info, "onnx")
    input_data = aipu_testing.get_op_input(
        model_name, input_shapes, {"from_type": from_type, "to_type": to_type}
    )
    aipu_output = aipu_testing.get_tvm_output(cfg_file, input_data)
    aipu_testing.get_test_result(aipu_testing.ONNXModel(cfg_file), input_data, aipu_output, 0.99)
