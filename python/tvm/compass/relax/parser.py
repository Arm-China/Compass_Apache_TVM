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


def parse_tensorflow(model_path, inputs, input_shape, outputs):
    """Simple wrapper of Relax Tensorflow API."""
    with open(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    with graph.as_default():
        # Import the graph definition into the current default graph. Even
        # through without this statement the TensorFlow model can be imported to
        # Relax correctly, but it is necessary, because it can report error if
        # the TensorFlow model is broken, besides that it is needed if we want
        # run this model in TensorFlow.
        tf.import_graph_def(graph_def, name="")

    from tvm.relax.frontend.tensorflow import from_tensorflow

    if not inputs:
        inputs = [n.name for n in graph.get_operations() if n.type == "Placeholder"]
    inputs = [inp[:-2] if inp[-2:] == ":0" else inp for inp in inputs]
    if not input_shape:
        input_shape = []
        for n in inputs:
            node = graph.get_operation_by_name(n)
            shape = node.outputs[0].shape
            try:
                if shape.rank is None:
                    shape = []
                else:
                    shape = [int(i) for i in shape]
            except:
                raise RuntimeError(f"please set tensor {node.outputs[0].name} shape in config file")
            input_shape.append(shape)
    shape_dict = {}
    for inp, shape in zip(inputs, input_shape):
        shape_dict[inp] = shape
    return from_tensorflow(graph_def, shape_dict=shape_dict, outputs=outputs)


def parse_tflite(model_path, inputs, input_shape):
    """Simple wrapper of Relax tflite API."""
    from tvm.relax.frontend.tflite import from_tflite

    with open(model_path, "rb") as f:
        # Get TFLite model from buffer
        try:
            import tflite

            tflite_model = tflite.Model.GetRootAsModel(f.read(), 0)
        except AttributeError:
            import tflite.Model

            tflite_model = tflite.Model.Model.GetRootAsModel(f.read(), 0)

    shape_dict = {}
    for inp, shape in zip(inputs, input_shape):
        shape_dict[inp] = shape
    return from_tflite(tflite_model, shape_dict)


def parse_model(parser_config):
    """Simple wrapper to call Relax frontend API."""
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
    elif framework in ["tensorflow", "tf"]:
        ir_mod = parse_tensorflow(model_path, inputs, input_shape, outputs)
    elif framework == "tflite":
        ir_mod = parse_tflite(model_path, inputs, input_shape)
    else:
        raise RuntimeError(f'Compass TVM parser can not support framework "{framework}" now.')
    return ir_mod
