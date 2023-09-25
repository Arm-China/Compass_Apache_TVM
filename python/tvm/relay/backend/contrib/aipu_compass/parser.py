# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""Simple wrap each Relay frontend API."""
# pylint: disable=import-outside-toplevel
import os
import re
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata

# Import TensorFlow locally in function will lead resource leak.
import tensorflow as tf

try:
    # Package "tf.compat.v1" is added from version "r1.13".
    tf_compat_v1 = tf.compat.v1  # pylint: disable=invalid-name
except AttributeError:
    tf_compat_v1 = tf  # pylint: disable=invalid-name


def parse_relay(model_path):
    with open(model_path) as f:
        return relay.fromtext(f.read())


def parse_tensorflow(model_path, inputs, input_shape, outputs):
    """Simple wrapper of Relay Tensorflow API."""
    with open(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    with graph.as_default():
        # Import the graph definition into the current default graph. Even
        # through without this statement the TensorFlow model can be imported to
        # Relay correctly, but it is necessary, because it can report error if
        # the TensorFlow model is broken, besides that it is needed if we want
        # run this model in TensorFlow.
        tf.import_graph_def(graph_def, name="")

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
                raise RuntimeError(
                    "please set tensor %s shape in config file" % node.outputs[0].name
                )
            input_shape.append(shape)
    shape_dict = {}
    for inp, shape in zip(inputs, input_shape):
        shape_dict[inp] = shape
    return relay.frontend.from_tensorflow(graph_def, shape=shape_dict, outputs=outputs)


def parse_onnx(model_path, inputs, input_shape):
    """Simple wrapper of Relay onnx API."""
    import onnx

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
    return relay.frontend.from_onnx(model, shape=shape_dict)


def parse_tflite(model_path, inputs, input_shape):
    """Simple wrapper of Relay tflite API."""
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
    return relay.frontend.from_tflite(tflite_model, shape_dict)


def parse_caffe(blob_path, proto_path, inputs, input_shape, dtype_dict=None):
    """Simple wrapper of Relay Caffe API."""

    assert os.path.isfile(blob_path), f"{blob_path} is not exist."
    assert os.path.isfile(proto_path), f"{proto_path} is not exist."

    from google.protobuf import text_format
    from caffe.proto import caffe_pb2 as pb

    init_net = pb.NetParameter()
    predict_net = pb.NetParameter()

    with open(blob_path, "rb") as f:
        init_net.ParseFromString(f.read())
    with open(proto_path, "r") as f:
        text_format.Merge(f.read(), predict_net)

    if not inputs:
        import caffe

        # Load Caffe model
        try:
            net = caffe.Net(proto_path, blob_path, caffe.TEST)
        except:
            raise RuntimeError("Caffe API parse Error happened.")
        # Generate input name list
        inputs = net.inputs

    assert len(inputs) == len(input_shape)
    shape_dict = {}
    for inp, shape in zip(inputs, input_shape):
        shape_dict[inp] = shape
    dtype_dict = dtype_dict or {}

    return relay.frontend.from_caffe(init_net, predict_net, shape_dict, dtype_dict)


def parse_darknet(cfg, weights, input_shape):
    """Simple wrapper of Relay darknet API."""
    if weights is None:
        raise RuntimeError("cannot parse darknet without weights")

    # Download libdarknet2.0.so
    darknet_lib_url = (
        "https://github.com/dmlc/web-data/blob/main/darknet/lib/libdarknet2.0.so?raw=true"
    )
    lib_path = download_testdata(darknet_lib_url, "libdarknet2.0.so", module="lib")

    # Load cfg and weights
    from tvm.relay.testing.darknet import __darknetffi__

    darknet_lib = __darknetffi__.dlopen(lib_path)
    net = darknet_lib.load_network(cfg.encode("utf-8"), weights.encode("utf-8"), 0)

    return relay.frontend.from_darknet(net, input_shape[0])


def parse_torch(model_path, inputs, input_shape, input_dtype):
    """Simple wrapper of Relay torch API."""
    import torch

    inputs_info = []
    if input_dtype is not None:
        input_dtype = input_dtype.split(",")
        input_dtype = [dtype.strip() for dtype in input_dtype]
        for name, shape, dtype in zip(inputs, input_shape, input_dtype):
            inputs_info.append(tuple(name, tuple(tuple(shape), dtype)))
    else:
        for name, shape in zip(inputs, input_shape):
            inputs_info.append((name, tuple(shape)))

    model = torch.jit.load(model_path)
    model.eval()
    return relay.frontend.from_pytorch(model, input_infos=inputs_info)


def parse_paddle(model_path, inputs, input_shape):
    """Simple wrapper of Relay paddle API."""
    assert os.path.isfile(model_path), f"{model_path} is not exist."
    assert len(inputs) == len(
        input_shape
    ), f"The number of names({len(inputs)}) and shapes({len(input_shape)}) should be same"

    input_shape_dict = {}
    for name, shape in zip(inputs, input_shape):
        input_shape_dict[name] = shape

    import paddle

    model_path = os.path.splitext(model_path)[0]
    model = paddle.jit.load(model_path)
    return relay.frontend.from_paddle(model, input_shape_dict)


def parse_model(parser_config):
    """Simple wrapper to call Relay frontend API."""
    ir_mod = tvm.IRModule()
    params = dict()

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
    if framework == "relay":
        ir_mod = parse_relay(model_path)
        return ir_mod, params
    if framework in ["tensorflow", "tf"]:
        ir_mod, params = parse_tensorflow(model_path, inputs, input_shape, outputs)
    elif framework == "onnx":
        ir_mod, params = parse_onnx(model_path, inputs, input_shape)
    elif framework == "darknet":
        weights = parser_config.get("weights", None)
        ir_mod, params = parse_darknet(model_path, weights, input_shape)
    elif framework == "tflite":
        ir_mod, params = parse_tflite(model_path, inputs, input_shape)
    elif framework == "caffe":
        prototxt = parser_config.get("caffe_prototxt", None)
        if prototxt is None:
            raise RuntimeError("caffe_prototxt also need be provied when parse caffe model")
        ir_mod, params = parse_caffe(model_path, prototxt, inputs, input_shape)
    elif framework == "torch":
        input_dtype = parser_config.get("input_dtype", None)
        ir_mod, params = parse_torch(model_path, inputs, input_shape, input_dtype)
    elif framework == "paddle":
        ir_mod, params = parse_paddle(model_path, inputs, input_shape)
    else:
        raise RuntimeError(
            f"AIPU Compass wrapper of TVM parser can't " f'support framework "{framework}" now.'
        )
    return ir_mod, params
