# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Custom operator plugin register for Compass."""
CODEGEN_CUSTOM_OP_DICT = dict()  # {op_name:op_type}
ONNX_CUSTOM_OP_DICT = dict()  # {op_type:op_converter}
TF_CUSTOM_OP_DICT = dict()  # {op_type:op_converter}


def codegen_plugin_register(op_name, op_type):
    CODEGEN_CUSTOM_OP_DICT[op_name] = op_type


def parser_plugin_register(model_type, op_name):
    """Regist op converter to parse custom op in tvm."""

    def register(func):
        if model_type.lower() == "onnx":
            ONNX_CUSTOM_OP_DICT[op_name] = func
        elif model_type.lower() == "tf":
            TF_CUSTOM_OP_DICT[op_name] = func
        else:
            raise RuntimeError(f"Unsupported model type {model_type}.")

        return func

    return register
