# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""Generate single-operator model files and configurations"""
import os
import random
import configparser
import numpy as np
import onnx
from ..parser import tf, tf_compat_v1
from .common import convert_to_list


POOLING_MAX_KERNEL = 65
CONV_MAX_KERNEL = 64
MAX_KERNEL_OUT_C = 100
MAX_RATE = 16  # Qian_SPEC 2,4,8,16
MAX_PAD = 6  # Qian_SPEC 6
MAX_STRIDE = 6  # Qian_SPEC 6
MAX_MULTIPLIER = 10


def get_input_tensor_of_tf(input_shapes, input_dtypes=None):
    """Get input tensor of tensorflow/tflite"""
    inputs_tensors = []
    for i, shape in enumerate(input_shapes):
        data_type = tf.float32
        if input_dtypes:
            data_type = input_dtypes[i]
        inputs = tf_compat_v1.placeholder(data_type, shape=shape)
        inputs_tensors.append(inputs)

    return inputs_tensors


def gen_dim_info(shapes):
    """The function to generate dim_info."""

    assert isinstance(shapes, list), "Only support list type."
    if isinstance(shapes[0], list):
        shapes = shapes[0]
    if shapes:
        return f"{len(shapes)}d"
    return "scalar"


def gen_model_name(op_type, *args):
    """The function to generate model_name."""

    assert isinstance(op_type, str), "The first arg must be string type"
    model_name_list = [op_type]

    def handle_single_ele(s_ele):
        if isinstance(s_ele, (int, float)):
            s_ele = str(s_ele).replace("-", "neg")
            s_ele = str(s_ele).replace(".", "dot")
        return str(s_ele).lower()

    def handle_single_list(s_list):
        return "-".join([handle_single_ele(x) for x in s_list])

    def handle_list(_list):
        if len(_list) == 0:
            return "scalar"
        if all([not isinstance(x, (list, tuple)) for x in _list]):
            return handle_single_list(_list)

        sub_name_list = []
        for ele in _list:
            if isinstance(ele, (list, tuple)):
                sub_name_list.append(handle_list(ele))
            else:
                sub_name_list.append(handle_single_ele(ele))
        return "_".join(sub_name_list)

    for arg in args:
        if isinstance(arg, (list, tuple)):
            model_name_list.append(handle_list(arg))
        else:
            model_name_list.append(handle_single_ele(arg))

    model_name = "_".join(model_name_list)
    return model_name


def gen_conv_params(
    inputs_shape: list, framework, group, depthwise=False, dilated=False
):  # pylint: disable=invalid-name
    """The function to generate params for conv op test."""

    _1d = False
    _2d = False
    _3d = False
    if len(inputs_shape) == 3:
        _1d = True
    elif len(inputs_shape) == 4:
        _2d = True
    elif len(inputs_shape) == 5:
        _3d = True
    else:
        raise RuntimeError("Only support length of input shape between 3 and 5.")

    if framework.lower() in ["onnx", "caffe"]:
        if _1d:
            _, c, w = inputs_shape
        if _2d:
            _, c, h, w = inputs_shape
        if _3d:
            _, c, d, h, w = inputs_shape
    else:
        if _1d:
            _, w, c = inputs_shape
        if _2d:
            _, h, w, c = inputs_shape
        if _3d:
            _, d, h, w, c = inputs_shape

    if _1d:
        dilation = random.randint(2, min(w, MAX_RATE)) if dilated else 1
    if _2d:
        dilation = random.randint(2, min(h, w, MAX_RATE)) if dilated else 1
    if _3d:
        dilation = random.randint(2, min(d, h, w, MAX_RATE)) if dilated else 1

    new_w = np.ceil(w / dilation)
    if _2d:
        new_h = np.ceil(h / dilation)
    if _3d:
        new_h = np.ceil(h / dilation)
        new_d = np.ceil(d / dilation)

    k_w = random.randint(1, min(new_w, CONV_MAX_KERNEL))
    kernel = [k_w]
    if _2d or _3d:
        k_h = random.randint(1, min(new_h, CONV_MAX_KERNEL))
        kernel = [k_h, k_w]
        if _3d:
            k_d = random.randint(1, min(new_d, CONV_MAX_KERNEL))
            kernel = [k_d, k_h, k_w]

    pad_w = random.randint(0, min(k_w, MAX_PAD) - 1)
    pad = [pad_w]
    if _2d or _3d:
        pad_h = random.randint(0, min(k_h, MAX_PAD) - 1)
        pad = [pad_h, pad_w]
        if _3d:
            pad_d = random.randint(0, min(k_d, MAX_PAD) - 1)
            pad = [pad_d, pad_h, pad_w]

    # Qian SPEC: if dilated stride must = 1
    s_w = random.randint(1, min(k_w, MAX_STRIDE)) if not dilated else 1
    stride = [s_w]
    if _2d or _3d:
        s_h = random.randint(1, min(k_h, MAX_STRIDE)) if not dilated else 1
        stride = [s_h, s_w]
        if _3d:
            s_d = random.randint(1, min(k_d, MAX_STRIDE)) if not dilated else 1
            stride = [s_d, s_h, s_w]

    if depthwise:
        num_output = random.randint(1, MAX_MULTIPLIER) * c
    else:
        num_output = random.randint(1, MAX_KERNEL_OUT_C) * group

    conv_params = {
        "dilation": dilation,
        "kernel": kernel,
        "pad": pad,
        "stride": stride,
        "num_output": num_output,
    }

    return conv_params


def get_pool_out_shapes(
    auto_pad, input_shapes, kernel_shape, strides, ceil_mode, pads, method, dilations="default"
):
    """Generate output shape for onnx's pooling operator."""
    spatial_shape_len = len(input_shapes[0]) - 2
    method = str(method).lower()
    assert method in ["avg", "max"], "Pooling method must be 'avg' or 'max'."
    assert (
        len(kernel_shape) == spatial_shape_len
    ), "The length of the kernel shape and spatial shape must be equal."
    ceil_mode = 0 if ceil_mode == "default" else ceil_mode
    pads = [0] * spatial_shape_len * 2 if pads == "default" else pads
    strides = [1] * spatial_shape_len if strides == "default" else strides
    dilations = [1] * spatial_shape_len if dilations == "default" else dilations

    pad_shape = []
    for i in range(spatial_shape_len):
        pad_shape.append(int(pads[i] + pads[i + spatial_shape_len]))
    padded_shapes = np.add(input_shapes[0][2:], pad_shape)

    out_shape = [0] * spatial_shape_len
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        for i in range(spatial_shape_len):
            out_shape[i] = int(np.ceil(float(padded_shapes[i]) / float(strides[i])))
    elif auto_pad == "VALID":
        if method == "avg":
            for i in range(spatial_shape_len):
                out_shape[i] = int(
                    np.ceil((float(padded_shapes[i] - kernel_shape[i] + 1)) / float(strides[i]))
                )
        else:
            for i in range(spatial_shape_len):
                out_shape[i] = int(
                    np.ceil(
                        (float(padded_shapes[i] - ((kernel_shape[i] - 1) * dilations[i] + 1) + 1))
                        / float(strides[i])
                    )
                )
    elif auto_pad in ("NOTSET", "default"):
        f = np.floor if ceil_mode == 0 else np.ceil
        if method == "avg":
            for i in range(spatial_shape_len):
                out_shape[i] = f((float(padded_shapes[i] - kernel_shape[i]) / strides[i] + 1))
        else:
            for i in range(spatial_shape_len):
                out_shape[i] = f(
                    (
                        float(padded_shapes[i] - ((kernel_shape[i] - 1) * dilations[i] + 1))
                        / strides[i]
                        + 1
                    )
                )
        out_shape = list(map(int, out_shape))

    output_shapes = [input_shapes[0][:2] + out_shape]
    return output_shapes


def skip_case(shape, axis, default_axis=None):
    """Skip op test cases that do not satisfy the condition."""
    r_length = len(shape[0]) if isinstance(shape[0], list) else len(shape)
    if axis in ["default", "random", None]:
        if default_axis is None or axis == "random":
            return False
        axis = default_axis
    if isinstance(axis, (list, tuple)):
        flag_skip = False
        for a in axis:
            if a not in range(-r_length, r_length):
                flag_skip = True
                break
        return flag_skip
    if axis not in range(-r_length, r_length):
        return True
    return False


def get_conv_in_out_shapes(
    input_shape,
    kernel_shape,
    strides,
    pads,
    group,
    dilations,
    auto_pad,
    num_output=10,
    output_padding="default",
    output_shape="default",
    is_bias=False,
    is_trans=False,
):  # pylint: disable=invalid-name
    """Generate the input and output shapes of convolution op."""
    spatial_length = len(input_shape) - 2
    assert len(kernel_shape) == spatial_length
    assert len(strides) == spatial_length
    assert len(pads) == spatial_length * 2
    assert len(dilations) == spatial_length

    if is_trans:
        num_output = num_output * group
        # (C x M/group x k1 x k2 x ... x kn)
        input_shape_weight = [input_shape[1], int(num_output / group)]
    else:
        # (M x C/group x k1 x k2 x ... x kn)
        input_shape_weight = [num_output, int(input_shape[1] / group)]

    for k in kernel_shape:
        input_shape_weight.append(int(k))

    weight_values = np.random.random(tuple(input_shape_weight)).astype(np.float32)
    input_shapes = [input_shape, input_shape_weight]
    default_values = [None, weight_values]
    if is_bias:
        input_shapes.append([num_output])
        bias_values = np.random.random(tuple([num_output])).astype(np.float32)
        default_values.append(bias_values)

    output_shapes = [input_shape[0], num_output]
    if output_shape != "default" and is_trans:
        assert len(output_shape) == spatial_length
        output_shapes += output_shape
    else:
        if auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
            if is_trans:
                # output_shape[i] = input_shape[i] * strides[i]
                for i, e in enumerate(input_shape[2:]):
                    output_shapes.append(int(e * strides[i]))
            else:
                # output_shape[i] = ceil(input_shape[i] / strides[i])
                for i, e in enumerate(input_shape[2:]):
                    output_shapes.append(int(np.ceil(float(e) / float(strides[i]))))
        elif auto_pad in ["default", "NOTSET", "VALID"]:
            if is_trans:
                if output_padding == "default":
                    output_padding = [0] * spatial_length
                # output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + \
                # ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]
                pads_half = len(pads) // 2
                for i, e in enumerate(input_shape[2:]):
                    shape_value = int(
                        strides[i] * (e - 1)
                        + output_padding[i]
                        + ((kernel_shape[i] - 1) * dilations[i] + 1)
                        - pads[i]
                        - pads[i + pads_half]
                    )
                    output_shapes.append(shape_value)
            else:
                for i, e in enumerate(input_shape[2:]):
                    shape_value = int(
                        (e + 2 * pads[i] - dilations[i] * (kernel_shape[i] - 1) - 1) / strides[i]
                        + 1
                    )
                    output_shapes.append(shape_value)
    output_shapes = [output_shapes]

    in_out_shapes_dict = {
        "input_shapes": input_shapes,
        "default_value": default_values,
        "output_shapes": output_shapes,
    }

    return in_out_shapes_dict


def gen_model(framework_name: str, model_info: dict, data_path: str):
    """Wrap the single OP into a model file."""
    if framework_name == "onnx":
        return _gen_onnx_model(model_info, data_path)

    if framework_name in ["tf", "tflite"]:
        return _gen_tf_model(model_info, data_path, framework_name)

    if framework_name == "caffe":
        return _gen_caffe_model(model_info, data_path)

    raise RuntimeError(f"Not support creating models of {framework_name}")


def _gen_onnx_model(model_info: dict, data_path: str):
    """Generate ONNX model."""
    # Check if variables that all we need is in model_info dict.
    list_keys = ["model_name", "op_type", "inputs", "outputs", "attributes", "opset"]
    is_all_true = [key in model_info for key in list_keys]
    if not all(is_all_true):
        for i, k in enumerate(is_all_true):
            if not k:
                print(f"{list_keys[i]} is not in model_info.")
        raise RuntimeError("ONNX: model_info is not valid.")

    model_name = model_info["model_name"]
    op_type = model_info["op_type"]
    inputs_info = model_info["inputs"]
    outputs_info = model_info["outputs"]
    attributes = model_info["attributes"]
    opset = model_info["opset"]

    def _check_model(model_graph):
        try:
            onnx.checker.check_model(model_graph)
        except onnx.checker.ValidationError as error_info:
            print("The model is invalid: %s" % error_info)
            return False
        else:
            print("The model is valid!")
            return True

    # Set name, tensor and data type for inputs
    input_names = []
    input_tensors = []
    initializer = []
    if str(model_name).split("_")[0].lower() != "constant":
        for i, shape in enumerate(inputs_info["shapes"]):
            if "data_types" in inputs_info.keys():
                data_type = inputs_info["data_types"][i]
            else:
                data_type = onnx.TensorProto.FLOAT
            if shape == "_":
                input_name = ""
            else:
                if (
                    "default_value" in inputs_info.keys()
                    and inputs_info["default_value"][i] is not None
                ):
                    input_name = f"Placeholder_{i}_Initializer"
                    tensor_value = inputs_info["default_value"][i]
                    if isinstance(tensor_value, np.ndarray):
                        tensor_value = tensor_value.flatten()
                    inp_tensor = onnx.helper.make_tensor(
                        name=input_name,
                        data_type=data_type,
                        dims=shape,
                        vals=tensor_value,
                        raw=False,
                    )
                    initializer.append(inp_tensor)
                else:
                    input_name = f"Placeholder_{i}"
                input_tensors.append(
                    onnx.helper.make_tensor_value_info(input_name, elem_type=data_type, shape=shape)
                )
            input_names.append(input_name)

    # Set name, tensor and data type for outputs
    output_names = []
    output_tensors = []
    for i, shape in enumerate(outputs_info["shapes"]):
        if shape == "_":
            output_name = ""
        else:
            output_name = f"{op_type}_{i}"
        output_names.append(output_name)
        if "data_types" in outputs_info.keys():
            data_type = outputs_info["data_types"][i]
        else:
            data_type = onnx.TensorProto.FLOAT
        if output_name != "":
            output_tensors.append(
                onnx.helper.make_tensor_value_info(output_name, elem_type=data_type, shape=shape)
            )

    # Generate node
    node = onnx.helper.make_node(op_type, inputs=input_names, outputs=output_names, **attributes)
    nodes = [node]

    # Generate graph
    graph = onnx.helper.make_graph(
        nodes,
        name=model_name,
        inputs=input_tensors,
        outputs=output_tensors,
        initializer=initializer,
    )

    # Generate model with opset version
    if not opset:
        onnx_model = onnx.helper.make_model(graph)
    else:
        opset = convert_to_list(opset)
        onnx_opsets = []
        for i in opset:
            onnx_opsets.append(onnx.helper.make_opsetid("", int(i)))
        onnx_model = onnx.helper.make_model(graph, opset_imports=onnx_opsets)

    # print('The model is:\n{}'.format(onnx_model))
    assert _check_model(onnx_model) is True, "The model is invalid."

    # Save model file
    model_path = os.path.join(data_path, model_name + ".onnx")
    if os.path.isfile(model_path):
        os.remove(model_path)
    onnx.save(onnx_model, model_path)
    os.system(f"chmod 777 {model_path}")
    return model_path


def _gen_tf_model(model_info: dict, data_path: str, framework: str):
    """Generate TF/TFLite model."""
    # Check if variables that all we need is in model_info dict.
    list_keys = ["model_name", "op_type", "inputs", "outputs", "in_graph"]
    is_all_true = [key in model_info for key in list_keys]
    if not all(is_all_true):
        for i, k in enumerate(is_all_true):
            if not k:
                print(f"{list_keys[i]} is not in model_info.")
        raise RuntimeError("TF/TFLite: model_info is not valid.")

    model_name = model_info["model_name"]
    op_type = model_info["op_type"]
    input_tensors = model_info["inputs"]
    output_tensors = model_info["outputs"]
    in_graph = model_info["in_graph"]

    # Wrap output_tensors
    output_tensors = convert_to_list(output_tensors)

    # Convert to list for some tflite operations
    if isinstance(output_tensors, tuple):
        output_tensors = list(output_tensors)

    # Generate model path
    if framework == "tf":
        f_name = f"{model_name}.pb"
    elif framework == "tflite":
        f_name = f"{model_name}.tflite"
    else:
        raise RuntimeError(f"Framework: {framework} is not valid.")

    model_path = os.path.join(data_path, f_name)
    if os.path.isfile(model_path):
        os.remove(model_path)

    # Generate model file
    with tf_compat_v1.Session(graph=in_graph) as sess:
        sess.run(tf_compat_v1.global_variables_initializer())

        output_nodes = []
        for t in output_tensors:
            node_name = t.name[:-2]
            if node_name in [n.name for n in sess.graph_def.node]:
                output_nodes.append(node_name)
            else:
                raise RuntimeError(f"Node: {node_name} not in Graph")

        output_nodes = list(set(output_nodes))

        if framework == "tf":
            if "CTCGreedy" in op_type:
                output_nodes = ["SparseToDense"]

            # Convert variables to constants
            frozen_graph_def = tf_compat_v1.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, output_nodes
            )

            # Save the frozen graph
            with open(model_path, "wb") as f:
                f.write(frozen_graph_def.SerializeToString())

        elif framework == "tflite":
            converter = tf_compat_v1.lite.TFLiteConverter.from_session(
                sess, input_tensors, output_tensors
            )

            # converter.allow_custom_ops = True
            # After TF 2.3, tflite can support TF ops
            # converter.target_spec.supported_ops = [
            #     tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            #     tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
            # ]
            tflite_model = converter.convert()
            with open(model_path, "wb") as f:
                f.write(tflite_model)

        os.system(f"chmod 777 {model_path}")
        return model_path


def _gen_caffe_model(model_info: dict, data_path: str):
    """Generate Caffe model."""
    # Check if variables that all we need is in model_info dict.
    list_keys = ["model_name", "net", "weights_dict"]
    is_all_true = [key in model_info for key in list_keys]
    if not all(is_all_true):
        for i, k in enumerate(is_all_true):
            if not k:
                print(f"{list_keys[i]} is not in model_info.")
        raise RuntimeError("Caffe: model_info is not valid.")

    model_name = model_info["model_name"]
    net = model_info["net"]
    weights_dict = model_info["weights_dict"]

    # Generate proto/model path
    model_path = os.path.join(data_path, model_name + ".caffemodel")
    proto_path = os.path.join(data_path, model_name + ".prototxt")
    if os.path.isfile(model_path):
        os.remove(model_path)
    if os.path.isfile(proto_path):
        os.remove(proto_path)

    # Generate proto file
    with open(proto_path, "w") as f:
        f.write(str(net.to_proto()))

    if not os.path.isfile(proto_path):
        raise FileNotFoundError(f"Gen {model_name} caffe prototxt Failed.")
    os.system(f"chmod 777 {proto_path}")

    # Suprress Caffe verbose prints
    os.environ["GLOG_minloglevel"] = "2"
    import caffe  # pylint: disable=import-outside-toplevel

    # Generate model file
    n = caffe.Net(proto_path, caffe.TRAIN)

    for key in weights_dict:
        if "weights" in weights_dict[key]:
            n.params[key][0].data.flat = weights_dict[key]["weights"]
        elif "mean" in weights_dict[key]:
            n.params[key][0].data.flat = weights_dict[key]["mean"]
            n.params[key][1].data.flat = weights_dict[key]["var"]
            if "scale" in weights_dict[key]:
                n.params[key][2].data.flat = weights_dict[key]["scale"]
        elif "scale" in weights_dict[key]:
            n.params[key][0].data.flat = weights_dict[key]["scale"]
        if "bias" in weights_dict[key]:
            n.params[key][1].data.flat = weights_dict[key]["bias"]
        if "gamma" in weights_dict[key]:  # used for prelu, not sure if other layers use this too
            n.params[key][0].data.flat = weights_dict[key]["gamma"]
    n.save(model_path)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Gen {model_name} caffe model Failed.")
    os.system(f"chmod 777 {model_path}")

    return model_path


def gen_cfg(
    model_type: str,
    model_name: str,
    model_path: str,
    input_shapes: list,
):
    """Generate cfg file for OP model."""

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"{model_path} is not exists.")

    config = configparser.ConfigParser()

    # Set "Common" section.
    config["Common"] = {
        "forward_engine": "opt_float",
        "executor": "vm",
    }

    # Set "Parser" section.
    if model_type in ["tf", "tensorflow"]:
        model_type = "tensorflow"
    else:
        model_type = model_type.lower()

    config["Parser"] = {
        "model_type": model_type,
        "model_name": model_name,
        "input_model": model_path,
        "input_shape": f"{input_shapes}",
    }
    if model_type == "caffe":
        caffe_prototxt = model_path.rsplit(".", 1)[0] + ".prototxt"
        config["Parser"]["caffe_prototxt"] = caffe_prototxt

    return config


def _check_if_model_exists(data_path, model_name, model_type):
    # Force the creation of a new model file.
    if os.environ.get("AIPU_TVM_GEN_MODEL", None) == "True":
        return False, None

    if model_type == "tf":
        f_name_list = [f"{model_name}.pb"]
    elif model_type == "tflite":
        f_name_list = [f"{model_name}.tflite"]
    elif model_type == "onnx":
        f_name_list = [f"{model_name}.onnx"]
    elif model_type == "caffe":
        f_name_list = [f"{model_name}.caffemodel", f"{model_name}.prototxt"]
    else:
        raise RuntimeError(f"Framework: {model_type} is not valid.")

    file_path_list = [os.path.join(data_path, f_name) for f_name in f_name_list]
    if all([os.path.isfile(f_path) for f_path in file_path_list]):
        return True, file_path_list[0]
    return False, None


def _check_if_cfg_exists(data_path, model_name):
    # Force the creation of a new cfg file.
    if os.environ.get("AIPU_TVM_GEN_CFG", None) == "True":
        return False, None

    cfg_path = os.path.join(data_path, model_name + ".cfg")
    if os.path.isfile(cfg_path):
        return True, cfg_path
    return False, None


def get_model_cfg_path(model_info: dict, model_type: str, need_cfg_str=True):
    """Get cfg path of model.

    Args:
        model_info (dict): dict to generate op model
        model_type (str): Original framework, e.g caffe
        need_cfg_str (bool): Get the content of cfg instead of the path of cfg

    Returns:
        str: A file path of cfg or the content of cfg, depends on the parameter need_cfg_str
    """
    for key in ["op_type", "model_name", "input_shapes"]:
        assert key in model_info

    model_name = model_info["model_name"]

    op_data_dir = os.environ.get(
        "AIPU_TVM_OP_DIR", os.path.abspath(f"{__file__}/../../../../../../../../op_files")
    )
    op_data_path = os.path.join(op_data_dir, model_info["op_type"], model_type)
    if not os.path.exists(op_data_path):
        original_umask = os.umask(0)
        try:
            os.makedirs(op_data_path, mode=0o777, exist_ok=True)
        finally:
            os.umask(original_umask)

    # If the model file already exists, return the model path
    # directly without creating a new one.
    is_model_exists, model_path = _check_if_model_exists(op_data_path, model_name, model_type)
    if not is_model_exists:
        # Generate the model file.
        model_path = gen_model(model_type, model_info, op_data_path)

    config = gen_cfg(
        model_type,
        model_name,
        model_path,
        model_info["input_shapes"],
    )

    if need_cfg_str:
        cfg_str_list = []
        for section in config.sections():
            cfg_str_list.append("[" + section + "]")
            for k, v in config[section].items():
                cfg_str_list.append(f"{k} = {v}")
        cfg_str = "\n".join(cfg_str_list)

        return cfg_str

    # If the cfg file already exists, return the cfg path
    # directly without creating a new one.
    is_cfg_exists, cfg_path = _check_if_cfg_exists(op_data_path, model_name)
    if not is_cfg_exists:
        # Clear the old file.
        cfg_path = os.path.join(op_data_path, model_name + ".cfg")
        if os.path.isfile(cfg_path):
            os.remove(cfg_path)
        # Write to file.
        with open(cfg_path, "w") as f:
            config.write(f)
        os.system(f"chmod 777 {cfg_path}")

    return cfg_path
