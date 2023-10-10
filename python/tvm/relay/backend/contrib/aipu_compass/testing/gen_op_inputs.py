# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
"""Generate the input data for OP test."""
import numpy as np
from onnx import TensorProto


UINT_OPS = ["Exp", "MaxPool", "OneHot", "Log"]
BOOL_OPS = ["LogicalAnd", "LogicalOr", "LogicalXor"]  # for ONNX
ZERO_ONE_OPS = [
    "ReduceAll",
    "ReduceAny",
    "LogicalAnd",
    "LogicalNot",
    "LogicalXor",
    "LogicalOr",
]  # for TF/TFlite

NP_DTYPE_MAPPING = {
    "bool": np.bool_,
    "float": np.float32,
    "float32": np.float32,
    "float16": np.float16,
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
}

ONNX_DTYPE_MAPPING = {
    "bool": TensorProto.BOOL,
    "float32": TensorProto.FLOAT,
    "float16": TensorProto.FLOAT16,
    "float": TensorProto.FLOAT,
    "bfloat16": TensorProto.BFLOAT16,
    "int8": TensorProto.INT8,
    "uint8": TensorProto.UINT8,
    "int16": TensorProto.INT16,
    "uint16": TensorProto.UINT16,
    "int32": TensorProto.INT32,
    "uint32": TensorProto.UINT32,
    "int64": TensorProto.INT64,
    "uint64": TensorProto.UINT64,
}


def gen_no_zero_data(shape):
    x = gen_pos_data(shape)
    y = np.random.choice([-1, 1], size=shape)
    return np.array(x * y)


def gen_pos_data(shape, low=1.0, high=50.0, dtype="float32"):
    dtype = NP_DTYPE_MAPPING[dtype.lower()]
    return np.random.uniform(low=low, high=high, size=tuple(shape)).astype(dtype)


def gen_uint_data(shape, low=1, high=100):
    return np.random.randint(low=low, high=high, size=tuple(shape))


def gen_int_data(shape, low=-50, high=100):
    return np.random.randint(low=low, high=high, size=tuple(shape))


def gen_common_data(shape, low=-10.0, high=50.0):
    return np.random.uniform(low=low, high=high, size=tuple(shape)).astype(np.float32)


def get_op_input(model_name, input_shapes, other_args=None, op_framework="onnx"):
    """Generate input data for OP model test."""
    if other_args is None:
        other_args = {}

    print(f"model name: {model_name}")
    op_type = model_name.split("_").pop(0)
    print(f"op type: {op_type}")

    input_data = []
    if len(input_shapes) >= 1 and (not isinstance(input_shapes[0], list)):
        input_shapes = [input_shapes]

    if op_type in UINT_OPS:
        dtype = "float32"
        if op_framework == "tf" and op_type == "OneHot":
            dtype = "int32"
        for shape in input_shapes:
            custom_data = gen_uint_data(shape, high=10).astype(dtype)
            input_data.append(custom_data)
    elif op_type in BOOL_OPS and op_framework == "onnx":
        for shape in input_shapes:
            custom_data = (np.random.uniform(low=-10.0, high=10.0, size=tuple(shape)) > 0).astype(
                np.bool_
            )
            input_data.append(custom_data)
    elif op_type == "Cast":
        in_dtype = str(other_args.get("from_type", "float32")).lower()
        out_dtype = str(other_args.get("to_type", "float32")).lower()
        for shape in input_shapes:
            if in_dtype.startswith("u") or out_dtype.startswith("u"):
                custom_data = gen_pos_data(shape)
            else:
                custom_data = gen_common_data(shape)
        input_data.append(custom_data)
    elif op_type == "Mod":
        in_dtype = str(other_args.get("from_type", "float32")).lower()
        for shape in input_shapes:
            input_data.append(gen_pos_data(shape, dtype=in_dtype))
    elif op_type == "BitShift":
        d_type = str(other_args.get("from_type", "float32")).lower()
        for shape in input_shapes:
            input_data.append(
                np.random.randint(low=0, high=20, size=tuple(shape)).astype(
                    NP_DTYPE_MAPPING[d_type]
                )
            )
    elif op_type in ["ScatterElements"]:
        for i, shape in enumerate(input_shapes):
            if i == 0:
                custom_data = gen_common_data(shape, high=10)
            elif i == 1:
                if "indices_data" in other_args:
                    custom_data = other_args["indices_data"]
                else:
                    custom_data = gen_common_data(shape, high=10)
            elif i == 2:
                custom_data = gen_common_data(shape, high=10)
            else:
                raise NotImplementedError
            input_data.append(custom_data)
    elif op_type == "ReverseSequence":
        for i, shape in enumerate(input_shapes):
            if i == 1:
                custom_data = other_args["seq_len"]
            else:
                custom_data = gen_common_data(shape)
            input_data.append(custom_data)
    elif op_type in ZERO_ONE_OPS and op_framework in ["tf", "tensorflow", "tflite"]:
        for input_shape in input_shapes:
            input_data.append(np.random.choice([0, 1], size=tuple(input_shape)).astype("float32"))
    else:
        print(" Using Random Data ".center(60, "="))
        for shape in input_shapes:
            input_data.append(gen_common_data(shape))
    return input_data
