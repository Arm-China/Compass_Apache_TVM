# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Implement the Compass IR generator of DSL program compilation."""
import os
import textwrap
import numpy as np


def _get_io_infos(param_infos, args):
    input_shapes, input_dtypes, output_shapes, output_dtypes = [], [], [], []

    for param_info, arg in zip(param_infos, args):
        if param_info.is_input_tensor or param_info.is_output_tensor:
            shapes, dtypes = input_shapes, input_dtypes
            if param_info.is_output_tensor:
                shapes, dtypes = output_shapes, output_dtypes

            shapes.append(list(arg.shape))
            dtypes.append(str(arg.dtype))

    return input_shapes, input_dtypes, output_shapes, output_dtypes


def gen_compass_ir(param_infos, args, op_type, ir_txt_path, ir_bin_path):
    """Generate the Compass IR for the corresponding DSL program."""
    assert len(args) == len(param_infos), "Argument count must equal to parameter's."
    input_shapes, input_dtypes, output_shapes, output_dtypes = _get_io_infos(param_infos, args)
    input_names = [f"input{i}" for i in range(len(input_shapes))]
    output_names = [f"output{i}" for i in range(len(output_shapes))]
    assert len(output_names) > 0, "Function must has at least one output."

    # 1. Generate The Header.
    layer_count = len(input_names) + 1
    ir_txt = textwrap.dedent(
        f"""\
        model_name=unknown
        layer_number={layer_count}
        data_format=NHWC
        precision=float32
        batch_size=1
        input_tensors=[{",".join(input_names)}]
        output_tensors=[{",".join(output_names)}]
        compat_quantized_model=false
        """
    )

    # 2. Generate All Input Layers.
    for i, (name, shape, dtype) in enumerate(zip(input_names, input_shapes, input_dtypes)):
        ir_txt += textwrap.dedent(
            f"""
            layer_id={i}
            layer_name={i}_input
            layer_type=Input
            layer_bottom=[]
            layer_bottom_shape=[]
            layer_bottom_type=[]
            layer_top=[{name}]
            layer_top_shape={shape}
            layer_top_type=[{dtype}]
            """
        )

    # 3. Generate The Layer That Represents the DSL Function.
    ir_txt += textwrap.dedent(
        f"""
        layer_id={layer_count - 1}
        layer_name=dsl_func
        layer_type={op_type}
        layer_bottom=[{",".join(input_names)}]
        layer_bottom_shape={input_shapes}
        layer_bottom_type=[{",".join(input_dtypes)}]
        layer_top=[{",".join(output_names)}]
        layer_top_shape={output_shapes}
        layer_top_type=[{",".join(output_dtypes)}]
        """
    )

    # Generate The Constant Data of the DSL Function as Weight.
    ir_bin = bytearray()
    for param_info, arg in zip(param_infos, args):
        if param_info.is_const_tensor:
            ir_txt += f"{param_info.name}_type={arg.dtype}\n"
            ir_txt += f"{param_info.name}_shape={list(arg.shape)}\n"
            ir_txt += f"{param_info.name}_offset={len(ir_bin)}\n"
            ir_txt += f"{param_info.name}_size={arg.nbytes}\n"
            ir_bin += arg.tobytes()

    ir_bin = np.frombuffer(ir_bin, dtype="uint8")

    # Generate The Scalar Values as Attribute.
    for param_info, arg in zip(param_infos, args):
        if param_info.is_attr:
            ir_txt += f"{param_info.name}={arg}\n"

    # 4. Write the Compass IR to disk.
    os.makedirs(os.path.dirname(ir_txt_path), exist_ok=True)
    open(ir_txt_path, "w", encoding="utf-8").write(ir_txt)
    os.makedirs(os.path.dirname(ir_bin_path), exist_ok=True)
    ir_bin.tofile(ir_bin_path)
