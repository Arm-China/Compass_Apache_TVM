# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Simple wrap each Relax frontend API."""
# pylint: disable=import-outside-toplevel
import re
import tvm
from tvm.script import from_source

# Import TensorFlow locally in function will lead resource leak.
import tensorflow as tf

try:
    # Package "tf.compat.v1" is added from version "r1.13".
    tf_compat_v1 = tf.compat.v1  # pylint: disable=invalid-name
except AttributeError:
    tf_compat_v1 = tf  # pylint: disable=invalid-name


def parse_relax(model_path):
    with open(model_path) as f:
        return from_source(f.read())


def parse_onnx(model_path, inputs, input_shape):
    """Simple wrapper of Relax onnx API."""
    import onnx
    from tvm.relax.frontend.onnx import from_onnx

    model = onnx.load(model_path)
    graph = model.graph

    def convert_name(n):
        for node in graph.node:
            if n == node.name:
                return node.output[0].name
        return n

    if not inputs:
        inputs = [i.name for i in graph.input]
    inputs = [convert_name(n) if n[-2:] != ":0" else n for n in inputs]
    if not input_shape:
        input_shape = []
        for i in graph.input:
            shape = []
            for dim in i.type.tensor_type.shape.dim:
                value = dim.dim_value
                if value is None or value == 0:
                    raise RuntimeError("please set tensor %s shape in config file" % i.name)
                shape.append(value)
            input_shape.append(shape)
    shape_dict = {}
    for inp, shape in zip(inputs, input_shape):
        shape_dict[inp] = shape
    return from_onnx(model, shape_dict=shape_dict)


def parse_model(parser_config):
    """Simple wrapper to call Relay frontend API."""
    ir_mod = tvm.IRModule()

    framework = parser_config["model_type"].lower()
    model_path = parser_config["input_model"]
    input_shape = parser_config.get("input_shape", None)
    inputs = parser_config.get("input", None)
    outputs = parser_config.get("output", None)
    if input_shape:
        input_shape = re.findall(r"\[[\s*\d+,]*\d+\]|\[\s*\]", input_shape)
        input_shape = [[int(i) for i in re.findall(r"\d+", shape)] for shape in input_shape]
    if inputs:
        inputs = [inp.strip() for inp in inputs.strip("[]").split(",")]
    if outputs:
        outputs = [out.strip() for out in outputs.strip("[]").split(",")]

    if framework == "relax":
        return parse_relax(model_path)

    if framework == "onnx":
        ir_mod = parse_onnx(model_path, inputs, input_shape)
    else:
        raise RuntimeError(
            f"AIPU Compass wrapper of TVM parser can't " f'support framework "{framework}" now.'
        )
    return ir_mod
