# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
""""Perform forward inference of the model through Original Framework."""
# pylint: disable=import-outside-toplevel
import os
import re
import configparser
import tvm
from tvm import relax
from ..utils import convert_to_tuple
from ..parser import tf, tf_compat_v1
from .common import convert_to_list, lower_framework


def _convert_to_tensor_name(name):
    """Convert to tensor name."""
    if not name.endswith(":0"):
        name += ":0"
    return name


def _print_framework_version(model_type):
    if model_type.lower() in ["tf", "tflite"]:
        framework_version = tf.__version__
    elif model_type.lower() == "onnx":
        import onnx

        framework_version = onnx.__version__
    elif model_type.lower() == "torch":
        import torch

        framework_version = torch.__version__
    elif model_type.lower() == "relax":
        framework_version = tvm.__version__
    else:
        print(f"Invalid Model Type: {model_type}, Can't Display Framework Version.")
        return
    print(f"{model_type} version: {framework_version}")


class Model(object):
    """Simple wrapper of Third-party framework forward flow."""

    def __init__(self, model_cfg):
        cfg_str = model_cfg
        cfg_dir = None
        if model_cfg.split(".")[-1] == "cfg":
            if not os.path.exists(model_cfg):
                raise FileNotFoundError(f"{model_cfg} does not exist.")
            cfg_dir = os.path.dirname(os.path.abspath(model_cfg))
            with open(model_cfg) as f:
                cfg_str = f.read()

        cfg_parser = configparser.ConfigParser()
        cfg_parser.read_string(cfg_str)

        parser = dict(cfg_parser["Parser"]) if "Parser" in cfg_parser else dict()
        if cfg_dir:
            for key, value in parser.items():
                parser[key] = value.replace("__CURRENT_CFG_DIR__", cfg_dir)
        self.parser = parser

        self.input_tensor_names = (
            [in_val.strip() for in_val in parser["input"].split(",")] if "input" in parser else []
        )
        self.output_tensor_names = (
            [out_val.strip() for out_val in parser["output"].split(",")]
            if "output" in parser
            else []
        )

        if "input_shape" in parser:
            input_shapes = re.findall(r"\[[\s*\d+,]*\d+\]|\[\s*\]", parser["input_shape"])
            self.input_shapes = [
                [int(i) for i in re.findall(r"\d+", shape)] for shape in input_shapes
            ]
        else:
            self.input_shapes = []

        self.model_type = lower_framework(parser.get("model_type", "none").strip())
        _print_framework_version(self.model_type)

        self.model_path = os.path.abspath(
            os.path.expanduser(os.path.expandvars(parser.get("input_model", "none")))
        )

        self.model_name = parser.get("model_name", "none")
        if self.model_name == "none":
            # name from cfg file name
            if model_cfg.endswith("cfg") and os.path.isfile(model_cfg):
                cfg_file_name = model_cfg.split("/")[-1].split(".")[0]
                if cfg_file_name.startswith(self.model_type + "_"):
                    self.model_name = cfg_file_name.split("_", 1)[1]
                else:
                    self.model_name = cfg_file_name

    def run(self, input_data):
        raise NotImplementedError("Not support general forward method yet.")


class TFModel(Model):
    """The implentation of TensorFlow framework forward flow."""

    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        # Convert to tensor name.
        self.input_tensor_names = [
            _convert_to_tensor_name(in_name) for in_name in self.input_tensor_names
        ]
        self.output_tensor_names = [
            _convert_to_tensor_name(out_name) for out_name in self.output_tensor_names
        ]

    def run(self, input_data):
        input_data = convert_to_list(input_data)
        pb_out = {}

        def get_inputs_outputs(graph):
            ops = graph.get_operations()
            outputs_set = set(ops)
            inputs = []
            for op in ops:
                if len(op.inputs) == 0 and op.type != "Const":
                    inputs.append(op)
                else:
                    for input_tensor in op.inputs:
                        if input_tensor.op in outputs_set:
                            outputs_set.remove(input_tensor.op)
            outputs = list(outputs_set)
            input_tensor_names = [
                _convert_to_tensor_name(i.name.replace("import/", "")) for i in inputs
            ]
            output_tensor_names = [
                _convert_to_tensor_name(i.name.replace("import/", "")) for i in outputs
            ]
            return (input_tensor_names, output_tensor_names)

        assert os.path.isfile(self.model_path), f"{self.model_path} is not exists."
        pb_file = self.model_path
        tf_compat_v1.reset_default_graph()
        with tf_compat_v1.Session() as sess:
            with open(pb_file, "rb") as graph:
                graph_def = tf_compat_v1.GraphDef()
                graph_def.ParseFromString(graph.read())
                tf.import_graph_def(graph_def)

                # Set the input/output tensor names
                input_names, output_names = get_inputs_outputs(sess.graph)
                if not self.input_tensor_names:
                    self.input_tensor_names = input_names
                if not self.output_tensor_names:
                    self.output_tensor_names = output_names

                if len(self.input_tensor_names) != len(input_data):
                    raise RuntimeError("Input Data len != Input Tensor len")

                input_num = len(self.input_tensor_names)
                output_num = len(self.output_tensor_names)

                return_ele = []
                for i in range(input_num):
                    return_ele.append(self.input_tensor_names[i])

                for i in range(output_num):
                    return_ele.append(self.output_tensor_names[i])

                ret = tf.import_graph_def(graph_def, return_elements=return_ele)

                # Set the input data
                input_tensor = {}
                for i in range(input_num):
                    input_tensor.update({ret[i]: input_data[i]})

                # Run model
                net_output = sess.run(ret[input_num:], feed_dict=input_tensor)

                # Get the output data
                for i in range(output_num):
                    pb_output = net_output[i]
                    pb_out.update({return_ele[input_num:][i]: pb_output})

                pb_output_flatten = {}

                for k, v in pb_out.items():
                    pb_output_flatten[k] = v.flatten()

        return pb_output_flatten


class TFLiteModel(Model):
    """The implentation of TensorFlow Lite framework forward flow."""

    def run(self, input_data):
        input_data = convert_to_list(input_data)
        tflite_out = {}

        assert os.path.isfile(self.model_path), f"{self.model_path} is not exists."
        tflite_file = self.model_path

        with open(tflite_file, "rb") as f:
            tflite_model_buf = f.read()

        try:
            from tensorflow import lite as interpreter_wrapper
        except ImportError:
            from tensorflow.contrib import lite as interpreter_wrapper

        interpreter = interpreter_wrapper.Interpreter(model_content=tflite_model_buf)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        if len(input_details) != len(input_data):
            raise RuntimeError("Input Data len != Input Tensor len")

        for i, input_detail in enumerate(input_details):
            interpreter.resize_tensor_input(input_detail["index"], input_data[i].shape)
        interpreter.allocate_tensors()

        # Set input
        for i, input_detail in enumerate(input_details):
            interpreter.set_tensor(input_detail["index"], input_data[i])

        # Sets the specified output tensor
        output_tensors = []
        if self.output_tensor_names:
            for output_name in self.output_tensor_names:
                for t in output_details:
                    if t["name"] == output_name:
                        output_tensors.append(t)
                        break
            if len(output_tensors) == 0:
                raise RuntimeError("Cannot find output node in TFlite")

        # Run
        interpreter.invoke()

        # Get output
        if output_tensors:
            output_details = output_tensors
        self.output_tensor_names = []
        for i, output_detail in enumerate(output_details):
            tflite_output = interpreter.get_tensor(output_detail["index"])
            tflite_out.update({output_detail["name"]: tflite_output})
            self.output_tensor_names.append(output_detail["name"])

        tflite_output_flatten = {}

        for k, v in tflite_out.items():
            tflite_output_flatten[k] = v.flatten()

        return tflite_output_flatten


class ONNXModel(Model):
    """The implentation of ONNX framework forward flow."""

    def run(self, input_data):
        input_data = convert_to_list(input_data)
        input_dict = {}
        output_dict = {}

        # Load ONNX model
        assert os.path.isfile(self.model_path), f"{self.model_path} is not exists."
        import onnxruntime

        sess = onnxruntime.InferenceSession(self.model_path)

        # Get io info
        if not self.input_tensor_names:
            inputs_details = sess.get_inputs()
            for i in inputs_details:
                self.input_tensor_names.append(i.name)
        if not self.output_tensor_names:
            outputs_details = sess.get_outputs()
            for i in outputs_details:
                self.output_tensor_names.append(i.name)

        if len(self.input_tensor_names) != len(input_data):
            raise RuntimeError("Input Data len != Input Tensor len")

        # Generate the input data dict
        for i in range(len(self.input_tensor_names)):
            input_dict.update({self.input_tensor_names[i]: input_data[i]})

        # Run model
        outputs = sess.run(output_names=None, input_feed=input_dict)
        meta = sess.get_modelmeta()

        # Get the output data dict
        if meta.producer_name != "tf2onnx":
            for i in range(len(self.output_tensor_names)):
                _out = outputs[i]
                output_dict.update({self.output_tensor_names[i]: _out})
        else:
            for i, out in enumerate(outputs):
                _out = out.astype(float)
                output_dict.update({self.output_tensor_names[i]: _out})

        output_flatten = {}
        for k, v in output_dict.items():
            output_flatten[k] = v.flatten()

        return output_flatten


class TorchModel(Model):
    """The implentation of PyTorch framework forward flow."""

    def run(self, input_data):
        import torch

        input_data = convert_to_list(input_data)
        input_data = [torch.from_numpy(inp) for inp in input_data]
        output_dict = {}

        assert os.path.isfile(self.model_path), f"{self.model_path} is not exists."
        net = torch.jit.load(self.model_path)
        outputs = net(*input_data)
        for out_name, out_data in zip(self.output_tensor_names, outputs):
            output_dict[out_name] = out_data.detach().numpy().flatten()

        return output_dict


class RelaxModel(Model):
    """The implentation of TVM framework forward flow."""

    def run(self, input_data):
        input_data = convert_to_list(input_data)

        assert os.path.isfile(self.model_path), f"{self.model_path} is not exists."
        with open(self.model_path) as f:
            mod = tvm.script.from_source(f.read())
        outputs = relax.VirtualMachine(relax.build(mod, "llvm"), tvm.cpu())["main"](*input_data)
        outputs = convert_to_tuple(outputs)

        output_dict = {}
        self.output_tensor_names = [f"out{i}" for i in range(len(outputs))]
        for out_name, out_data in zip(self.output_tensor_names, outputs):
            output_dict[out_name] = out_data.numpy().flatten()

        return output_dict
