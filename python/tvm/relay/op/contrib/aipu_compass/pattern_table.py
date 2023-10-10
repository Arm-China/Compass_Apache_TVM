# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
# pylint: disable=unused-argument, unsupported-binary-operation
"""Relay IR to Compass IR mapping rules."""
import inspect
from functools import reduce, wraps
from operator import mul
from typing import List, Tuple, Union
import numpy as np

from tvm.ir.container import Array
from tvm.tir import Any
from tvm import relay, ir, tir
from tvm.relay.dataflow_pattern import (
    is_constant,
    is_op,
    wildcard,
    is_var,
    is_tuple_get_item,
    is_tuple,
)
from tvm.relay.backend.contrib.aipu_compass import AipuCompassConfig


TOTAL_SHAPE_SIZE = 64 * 1024 * 1024


def get_activation_str(output_scale, output_zp, clip):
    """Get the AIPU Compass "with_activation" value from Relay IR."""
    output_scale = float(output_scale.data.numpy())
    output_zp = int(output_zp.data.numpy())
    # Dequantize a quantized integer value to a float value.
    dequantize = lambda x: output_scale * (x - output_zp)
    min_value = dequantize(clip.attrs.a_min)
    max_value = dequantize(clip.attrs.a_max)
    if np.isclose(min_value, 0):
        if np.isclose(max_value, 6, 1.0e-2):
            return "RELU6"
        return "RELU"
    return None


def unpack_commutative_args(call, rhs_name="const"):
    """Unpack arguments of the binary operators consider commutative, ensure the
    right hand side operand is the expected one."""
    assert isinstance(call, relay.Call)

    lhs, rhs = call.args
    if rhs_name == "const":
        if isinstance(rhs, relay.Constant):
            return lhs, rhs
        assert call.op == relay.op.get("add") or call.op == relay.op.get("multiply")
        assert isinstance(lhs, relay.Constant)
        return rhs, lhs

    if isinstance(rhs, relay.Call) and rhs.op == relay.op.get(rhs_name):
        return lhs, rhs

    assert call.op == relay.op.get("add") or call.op == relay.op.get("multiply")
    assert isinstance(lhs, relay.Call) and lhs.op == relay.op.get(rhs_name)
    return rhs, lhs


def _is_dynamic(shape_or_type):
    def _get_shapes(ir_type):
        if isinstance(ir_type, ir.TensorType):
            return [ir_type.shape]
        if isinstance(ir_type, ir.TupleType):
            shapes = []
            for field in ir_type.fields:
                shapes += _get_shapes(field)
            return shapes
        return []

    shapes = _get_shapes(shape_or_type) if isinstance(shape_or_type, ir.Type) else [shape_or_type]
    for shape in shapes:
        if any(isinstance(dim, Any) for dim in shape):
            return True
    return False


def _checker(func):
    func_name = func.__name__

    @wraps(func)
    def _wrapper(call: relay.Call):
        if isinstance(call, relay.Tuple):
            in_types = []
            for ele in call:
                in_types += [x.checked_type for x in ele.args]
        else:
            in_types = [x.checked_type for x in call.args]
        types = in_types + [call.checked_type]
        if any(_is_dynamic(x) for x in types):
            return False

        ret = func(call)
        is_generator = inspect.isgenerator(ret)

        if is_generator:
            generator = ret
            try:
                # Execute the code before the "yield" statement.
                next(generator)
            except StopIteration as e:  # pylint: disable=invalid-name
                value = e.value
                assert isinstance(value, bool), f'The return type of "{func_name}" must be "bool".'
                return value

        # The code used to check the Compass OP Spec can be skipped by users,
        # for the generator function, they are the code after the "yield"
        # statement, otherwise, they are the whole function body.
        if AipuCompassConfig.get().common["disable_op_spec_checker"] == "true":
            return True

        if not is_generator:
            return ret

        try:
            # Execute the code after the "yield" statement.
            next(generator)
        except StopIteration as e:  # pylint: disable=invalid-name
            assert isinstance(e.value, bool), f'The return type of "{func_name}" must be "bool".'
            return e.value
        raise RuntimeError(f'There is more than one "yield" in "{func_name}".')

    return _wrapper


def _is_scalar_and_close(x, ref):
    assert isinstance(x, (relay.Constant, int, float))
    assert isinstance(ref, (int, float))
    if isinstance(x, relay.Constant):
        x = x.data.numpy()
    if x.size != 1:
        return False
    return np.isclose(float(x), float(ref))


def _check_dim(shapes: Union[List[Array], Tuple[Array]], dims: List[int], keep_dims=True) -> bool:
    shape_dims = set(len(shape) for shape in shapes)
    if not shape_dims.issubset(set(dims)):
        return False
    if keep_dims:
        return len(shape_dims) == 1
    return True


def _convert_array(arr):
    if isinstance(arr, (tir.IntImm, int)):
        return int(arr)
    if isinstance(arr, (list, Tuple, Array)):
        return [_convert_array(val) for val in arr]
    return int(arr)


def _check_total_shape_size(check_list: Union[List[int], Tuple[int], Array]) -> bool:
    # ensure that shape should not be dynamic.
    if _is_dynamic(check_list):
        return False
    # all the input/output tensors must keep shape_size <= 64 MB
    check_list_int = _convert_array(check_list)
    if isinstance(check_list_int, list):
        size = int(np.prod(check_list_int))
    else:
        size = check_list_int
    return size <= TOTAL_SHAPE_SIZE


def _check_range(
    check_list: Union[List[int], Tuple[int], Array], rule: List[int], min_value=1
) -> bool:
    if len(check_list) != len(rule):
        return False
    # ensure that shape should not be dynamic.
    if isinstance(check_list, (list, Tuple, Array)) and _is_dynamic(check_list):
        return False

    return all(min_value <= x <= y for x, y in zip(_convert_array(check_list), rule))


def _check_shape(
    shape: Union[List[int], Tuple[int]], shape_0_idx: int, shape_n_idx: int, largest_dim: int
) -> bool:
    if _is_dynamic(shape):
        return False

    # The size limit of dim 0
    shape_0_pat = (
        [32],
        [16],
    )
    # The size limit of dim n(dimensions other than the dim 0)
    shape_n_pat = (
        [16384] * 4,
        [1920, 1080, 1920, 4096],
        [100, 1080, 1920, 4096],
    )
    assert -len(shape_0_pat) <= shape_0_idx <= len(shape_0_pat) - 1
    assert -len(shape_n_pat) <= shape_n_idx <= len(shape_n_pat) - 1

    dim = len(shape)
    spec = (
        shape_0_pat[shape_0_idx]
        + shape_n_pat[shape_n_idx][-(largest_dim - 1) :][largest_dim - dim :]
    )
    return _check_range(shape, spec)


@ir.register_op_attr("qnn.quantize", "target.aipu_compass.qnn")
@ir.register_op_attr("qnn.dequantize", "target.aipu_compass.qnn")
@ir.register_op_attr("strided_slice", "target.aipu_compass.qnn")
@ir.register_op_attr("strided_slice", "target.aipu_compass")
@ir.register_op_attr("negative", "target.aipu_compass")
@ir.register_op_attr("meshgrid", "target.aipu_compass")
@ir.register_op_attr("contrib.aipu_compass.fake_quant_with_min_max_vars", "target.aipu_compass")
@ir.register_op_attr("cast", "target.aipu_compass")
@ir.register_op_attr("where", "target.aipu_compass")
@ir.register_op_attr("tile", "target.aipu_compass.qnn")
@ir.register_op_attr("tile", "target.aipu_compass")
@ir.register_op_attr("abs", "target.aipu_compass")
@ir.register_op_attr("exp", "target.aipu_compass.qnn")
@ir.register_op_attr("exp", "target.aipu_compass")
@ir.register_op_attr("log", "target.aipu_compass")
@ir.register_op_attr("cos", "target.aipu_compass")
@ir.register_op_attr("nn.relu", "target.aipu_compass")
@ir.register_op_attr("nn.leaky_relu", "target.aipu_compass")
@ir.register_op_attr("clip", "target.aipu_compass")
@ir.register_op_attr("clip", "target.aipu_compass.qnn")
@ir.register_op_attr("qnn.sigmoid", "target.aipu_compass.qnn")
@ir.register_op_attr("sigmoid", "target.aipu_compass")
@ir.register_op_attr("power", "target.aipu_compass")
@ir.register_op_attr("sqrt", "target.aipu_compass")
@ir.register_op_attr("qnn.rsqrt", "target.aipu_compass.qnn")
@ir.register_op_attr("rsqrt", "target.aipu_compass")
@ir.register_op_attr("sin", "target.aipu_compass")
@ir.register_op_attr("tan", "target.aipu_compass")
@ir.register_op_attr("qnn.tanh", "target.aipu_compass.qnn")
@ir.register_op_attr("tanh", "target.aipu_compass")
@ir.register_op_attr("reshape", "target.aipu_compass.qnn")
@ir.register_op_attr("reshape", "target.aipu_compass")
@ir.register_op_attr("greater", "target.aipu_compass")
@ir.register_op_attr("less", "target.aipu_compass")
@ir.register_op_attr("equal", "target.aipu_compass")
@ir.register_op_attr("not_equal", "target.aipu_compass")
@ir.register_op_attr("greater_equal", "target.aipu_compass")
@ir.register_op_attr("less_equal", "target.aipu_compass")
@ir.register_op_attr("logical_and", "target.aipu_compass")
@ir.register_op_attr("logical_or", "target.aipu_compass")
@ir.register_op_attr("logical_not", "target.aipu_compass")
@ir.register_op_attr("logical_xor", "target.aipu_compass")
@ir.register_op_attr("bitwise_and", "target.aipu_compass")
@ir.register_op_attr("bitwise_or", "target.aipu_compass")
@ir.register_op_attr("bitwise_not", "target.aipu_compass")
@ir.register_op_attr("bitwise_xor", "target.aipu_compass")
@ir.register_op_attr("qnn.add", "target.aipu_compass.qnn")
@ir.register_op_attr("add", "target.aipu_compass")
@ir.register_op_attr("subtract", "target.aipu_compass")
@ir.register_op_attr("qnn.subtract", "target.aipu_compass.qnn")
@ir.register_op_attr("qnn.mul", "target.aipu_compass.qnn")
@ir.register_op_attr("multiply", "target.aipu_compass.qnn")
@ir.register_op_attr("multiply", "target.aipu_compass")
@ir.register_op_attr("divide", "target.aipu_compass")
@ir.register_op_attr("maximum", "target.aipu_compass")
@ir.register_op_attr("minimum", "target.aipu_compass")
@ir.register_op_attr("minimum", "target.aipu_compass.qnn")
@ir.register_op_attr("contrib.aipu_compass.detection_output", "target.aipu_compass")
@ir.register_op_attr("contrib.aipu_compass.nms", "target.aipu_compass")
@ir.register_op_attr("contrib.aipu_compass.decode_box", "target.aipu_compass")
@_checker
def _check_nothing(call: relay.Call):
    return True


@ir.register_op_attr("qnn.requantize", "target.aipu_compass.qnn")
@_checker
def _check(call: relay.Call):
    attrs = call.attrs
    dim = len(call.args[0].checked_type.shape)
    if attrs.axis not in [-1, dim - 1]:
        return False

    yield  # The Compass OP Spec check code must be placed after this statement.

    res = []
    res.append(attrs.out_dtype in ["int8", "uint8", "int16", "uint16"])

    return all(res)


@ir.register_op_attr("prod", "target.aipu_compass")
@ir.register_op_attr("sum", "target.aipu_compass")
@ir.register_op_attr("variance", "target.aipu_compass")
@ir.register_op_attr("mean", "target.aipu_compass")
@ir.register_op_attr("min", "target.aipu_compass")
@ir.register_op_attr("max", "target.aipu_compass")
@ir.register_op_attr("any", "target.aipu_compass")
@ir.register_op_attr("all", "target.aipu_compass")
@_checker
def _check_reduce(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    if not call.attrs.keepdims:
        return False

    yield  # The Compass OP Spec check code must be placed after this statement.

    return True


def _softplus_pattern():
    exp = is_op("exp")(wildcard())
    add = is_op("add")(exp, is_constant())
    pattern = is_op("log")(add)

    @_checker
    def check(softplus: relay.Call):
        # Check if the given match is supported by AIPU Compass.
        _, add_const = unpack_commutative_args(softplus.args[0])
        if not _is_scalar_and_close(add_const, 1):
            return False
        yield  # The Compass OP Spec check code must be placed after this statement.
        return True

    return ("aipu_compass.Softplus", pattern, check)


def _check_conv2d(conv2d: relay.Call, add: relay.Call):
    # Check if it is supported by AIPU Compass.
    attrs = conv2d.attrs

    if add:
        _, constant = unpack_commutative_args(add)
        squeezed_shape = [x for x in constant.checked_type.shape if x != 1]
        if not squeezed_shape:
            squeezed_shape = [1]
        if len(squeezed_shape) != 1 or squeezed_shape[0] != attrs.channels:
            return False

    yield  # The Compass OP Spec check code must be placed after this statement.

    in_shape = conv2d.args[0].checked_type.shape
    out_shape = conv2d.checked_type.shape

    res = []
    # only support 4-dim
    res.append(_check_dim([in_shape, out_shape], [4]))
    # check shape of input and output
    res.append(_check_range(in_shape[1:], [16384, 16384, 16384]))
    res.append(_check_range(out_shape[1:], [16384, 16384, 16384]))
    # check kernel
    kernels = attrs.kernel_size
    res.append(_check_range(kernels, [64, 64]))
    # check stride
    strides = attrs.strides
    res.append(_check_range(strides, [16, 16]))
    # check pad
    # can actually support larger pads, just skip
    # pads = attrs.padding
    # res.append(_check_range(pads, [16, 16, 16, 16], 0))
    # check dilation
    dilation = attrs.dilation
    res.append(_check_range(dilation, [40, 40]))
    if dilation[0] != 1 or dilation[1] != 1:
        # Here need to multiply sizeof(output_type), but cannnot get quantified type.
        # So use 16bit as default.
        res.append(reduce(mul, out_shape) * reduce(mul, strides) * 2 <= 1024 * 1024 * 1024)

    return all(res)


@ir.register_op_attr("nn.conv2d", "target.aipu_compass")
@_checker
def _check(conv2d: relay.Call):
    # Check if it is supported by AIPU Compass.
    return _check_conv2d(conv2d, None)


def _convolution2d_pattern():
    conv = is_op("nn.conv2d")(wildcard(), is_constant()) | is_op("nn.conv2d_transpose")(
        wildcard(), is_constant()
    )
    conv_add = is_op("add")(conv, is_constant())
    conv_add_relu = is_op("nn.relu")(conv_add)
    conv_add_leaky_relu = is_op("nn.leaky_relu")(conv_add)
    conv_add_clip = is_op("clip")(conv_add).has_attr({"a_min": 0.0, "a_max": 6.0})
    conv_leaky_relu = is_op("nn.leaky_relu")(conv)
    conv_relu = is_op("nn.relu")(conv)

    pattern = (
        conv
        | conv_add
        | conv_add_relu
        | conv_add_leaky_relu
        | conv_add_clip
        | conv_leaky_relu
        | conv_relu
    )

    @_checker
    def check(call: relay.Call):
        add = None
        if conv.match(call):
            conv2d = call
        elif (
            conv_add_relu.match(call)
            or conv_add_clip.match(call)
            or conv_add_leaky_relu.match(call)
        ):
            activation = call
            add = activation.args[0]
            conv2d, _ = unpack_commutative_args(add)
        elif conv_add.match(call):
            add = call
            conv2d, _ = unpack_commutative_args(add)
        elif conv_leaky_relu.match(call) or conv_relu.match(call):
            activation = call
            conv2d = activation.args[0]

        return _check_conv2d(conv2d, add)

    return ("aipu_compass.Convolution2D", pattern, check)


def _qnn_basiclstm_pattern():
    # fc1
    qnn_dense = is_op("qnn.dense")(
        wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
    )
    qnn_dense_requant_fc1 = is_op("qnn.requantize")(
        qnn_dense, is_constant(), is_constant(), is_constant(), is_constant()
    )
    # fc2
    qnn_dense1 = is_op("qnn.dense")(
        wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
    )
    qnn_dense_requant_fc2 = is_op("qnn.requantize")(
        qnn_dense1, is_constant(), is_constant(), is_constant(), is_constant()
    )

    qnn_add_fc = is_op("qnn.add")(
        qnn_dense_requant_fc1,
        qnn_dense_requant_fc2,
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )

    qnn_add_bias = is_op("qnn.add")(
        qnn_add_fc,
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )

    split = is_op("split")(qnn_add_bias)
    split0 = is_tuple_get_item(split, 0)
    split1 = is_tuple_get_item(split, 1)
    split2 = is_tuple_get_item(split, 2)
    split3 = is_tuple_get_item(split, 3)

    sigmoid_sp0 = is_op("qnn.sigmoid")(
        split0, is_constant(), is_constant(), is_constant(), is_constant()
    )
    tanh_sp2 = is_op("qnn.tanh")(split2, is_constant(), is_constant(), is_constant(), is_constant())
    mul_sp01 = is_op("qnn.mul")(
        sigmoid_sp0,
        tanh_sp2,
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )

    sigmoid_sp1 = is_op("qnn.sigmoid")(
        split1, is_constant(), is_constant(), is_constant(), is_constant()
    )
    sigmoid_sp3 = is_op("qnn.sigmoid")(
        split3, is_constant(), is_constant(), is_constant(), is_constant()
    )

    mul_sp2 = is_op("qnn.mul")(
        sigmoid_sp1,
        wildcard(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )
    add_cout = is_op("qnn.add")(
        mul_sp2,
        mul_sp01,
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )
    tanh_c = is_op("qnn.tanh")(add_cout, is_constant(), is_constant(), is_constant(), is_constant())

    mul_hout = is_op("qnn.mul")(
        sigmoid_sp3,
        tanh_c,
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )
    pattern = is_tuple((mul_hout, add_cout))

    @_checker
    def check(call: relay.Call):
        mul_hout, add_cout = call
        mul_sp2 = add_cout.args[0]

        sigmoid_sp1, initial_c = mul_sp2.args[:2]

        split = sigmoid_sp1.args[0].tuple_value
        add_bias = split.args[0]
        add_fc = add_bias.args[0]

        req_after_squeeze, req_after_hstate = add_fc.args[:2]
        q_dense_after_squeeze = req_after_squeeze.args[0]
        q_dense_after_hstate = req_after_hstate.args[0]

        input_x = q_dense_after_squeeze.args[0]
        initial_h = q_dense_after_hstate.args[0]

        in_shape0 = input_x.checked_type.shape
        in_shape1 = initial_h.checked_type.shape
        in_shape2 = initial_c.checked_type.shape
        out_shape0 = mul_hout.checked_type.shape
        out_shape1 = add_cout.checked_type.shape

        res = []
        # only support 2-dim(in_shape0 will be reshaped to 3 dims in the codegen)
        res.append(_check_dim([in_shape0, in_shape1, in_shape2, out_shape0, out_shape1], [2]))
        # check shape of input
        res.append(_check_range(in_shape0[:], [3071, 3072]))
        res.append(_check_range(in_shape1[:], [32, 3072]))
        res.append(_check_range(in_shape2[:], [32, 3072]))

        return all(res)

    return ("aipu_compass.QnnBasicLSTM", pattern, check)


def _qnn_convolution2d_pattern():
    q_conv = is_op("qnn.conv2d")(
        wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
    ) | is_op("qnn.conv2d_transpose")(
        wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
    )
    q_conv_add = is_op("add")(q_conv, is_constant())
    q_conv_add_req = is_op("qnn.requantize")(
        q_conv_add, is_constant(), is_constant(), is_constant(), is_constant()
    )
    q_conv_add_req_clip = is_op("clip")(q_conv_add_req)
    q_conv_add_req_leaky_relu = is_op("qnn.leaky_relu")(
        q_conv_add_req, is_constant(), is_constant(), is_constant(), is_constant()
    )
    q_conv_add_req_leaky_relu_req = is_op("qnn.requantize")(
        q_conv_add_req_leaky_relu, is_constant(), is_constant(), is_constant(), is_constant()
    )
    q_conv_req = is_op("qnn.requantize")(
        q_conv, is_constant(), is_constant(), is_constant(), is_constant()
    )
    q_conv_req_leaky_relu = is_op("qnn.leaky_relu")(
        q_conv_req, is_constant(), is_constant(), is_constant(), is_constant()
    )
    q_conv_req_leaky_relu_req = is_op("qnn.requantize")(
        q_conv_req_leaky_relu, is_constant(), is_constant(), is_constant(), is_constant()
    )

    pattern = (
        q_conv_req
        | q_conv_add_req
        | q_conv_add_req_clip
        | q_conv_add_req_leaky_relu
        | q_conv_add_req_leaky_relu_req
        | q_conv_req_leaky_relu_req
    )

    @_checker
    def check(call: relay.Call):
        if q_conv_add_req_leaky_relu_req.match(call) or q_conv_req_leaky_relu_req.match(call):
            post_requantize = call
            leaky_relu = post_requantize.args[0]
            requantize = leaky_relu.args[0]
        elif q_conv_add_req_leaky_relu.match(call):
            leaky_relu = call
            requantize = leaky_relu.args[0]
        elif q_conv_add_req.match(call):
            requantize = call
        elif q_conv_add_req_clip.match(call):
            clip = call
            requantize = clip.args[0]
            if get_activation_str(requantize.args[3], requantize.args[4], clip) is None:
                return False
        elif q_conv_req.match(call):
            requantize = call

        if requantize.args[0].op == relay.op.get("add"):
            add = requantize.args[0]
            q_conv, _ = unpack_commutative_args(add)
        else:
            add = None
            q_conv = requantize.args[0]

        return _check_conv2d(q_conv, add)

    return ("aipu_compass.QnnConvolution2D", pattern, check)


@ir.register_op_attr("image.resize2d", "target.aipu_compass")
@ir.register_op_attr("image.resize2d", "target.aipu_compass.qnn")
@_checker
def _resize2d_check(resize2d: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape = resize2d.args[0].checked_type.shape
    attrs = resize2d.attrs

    res = []
    # only support 4-dim
    res.append(_check_dim([in_shape], [4]))
    # check shape of input
    res.append(_check_range(in_shape[1:], [1080, 1920, 16384]))
    # check method
    res.append(attrs.method in ["nearest_neighbor", "linear"])
    # check mode
    res.append(
        attrs.coordinate_transformation_mode
        in [
            "half_pixel",
            "align_corners",
            "asymmetric",
            "pytorch_half_pixel",
            "tf_half_pixel_for_nn",
        ]
    )

    return all(res)


def _convolution3d_pattern():
    conv3d = is_op("nn.conv3d")(wildcard(), is_constant())
    conv3d_add = is_op("add")(conv3d, is_constant())
    conv3d_add_relu = is_op("nn.relu")(conv3d_add)
    conv3d_add_leaky_relu = is_op("nn.leaky_relu")(conv3d_add)
    conv3d_add_clip = is_op("clip")(conv3d_add).has_attr({"a_min": 0.0, "a_max": 6.0})
    conv3d_leaky_relu = is_op("nn.leaky_relu")(conv3d)
    conv3d_relu = is_op("nn.relu")(conv3d)

    pattern = (
        conv3d
        | conv3d_add
        | conv3d_add_relu
        | conv3d_add_leaky_relu
        | conv3d_add_clip
        | conv3d_leaky_relu
        | conv3d_relu
    )

    @_checker
    def check(call: relay.Call):
        # Check if the given match is supported by AIPU Compass.
        if conv3d.match(call):
            conv3d_call = call
        elif (
            conv3d_add_relu.match(call)
            or conv3d_add_clip.match(call)
            or conv3d_add_leaky_relu.match(call)
        ):
            activation = call
            add = activation.args[0]
            conv3d_call, _ = unpack_commutative_args(add)
        elif conv3d_add.match(call):
            conv3d_call, _ = unpack_commutative_args(call)
        elif conv3d_leaky_relu.match(call) or conv3d_relu.match(call):
            conv3d_call = call.args[0]

        # only check conv3d
        in_shape = conv3d_call.args[0].checked_type.shape
        out_shape = conv3d_call.checked_type.shape
        attrs = conv3d_call.attrs

        res = []
        # only support 5-dim
        res.append(not _is_dynamic(in_shape))
        res.append(_check_dim([in_shape, out_shape], [5]))
        # check shape of input and output
        res.append(_check_range(in_shape[:], [32, 100, 1080, 1920, 4096]))
        res.append(_check_range(out_shape[1:], [100, 1080, 1920, 4096]))
        # check kernel
        kernels = attrs.kernel_size
        res.append(_check_range(kernels, [11, 11, 11]))
        res.append(_check_range([in_shape[4] * kernels[0]], [4096]))
        # check stride
        strides = attrs.strides
        if _check_range(strides, [min(6, x) for x in kernels]) or (
            all(x == 1 for x in kernels) and all(y == 2 for y in strides)
        ):
            res.append(True)
        else:
            res.append(False)
        # check pad
        res.append(
            _check_range(
                attrs.padding,
                [max(6, kernels[0])] * 2 + [max(6, kernels[1])] * 2 + [max(6, kernels[2])] * 2,
                0,
            )
        )
        # check dilation.
        res.append(all(x == 1 for x in attrs.dilation))
        res.append(attrs.groups == 1)

        return all(res)

    return ("aipu_compass.Convolution3D", pattern, check)


def _elementwise_relu_pattern():
    pattern_op = is_op("add") | is_op("subtract") | is_op("multiply")
    pattern = pattern_op(wildcard(), wildcard())
    pattern = is_op("nn.relu")(pattern)

    return ("aipu_compass.ElementwiseRelu", pattern, _check_nothing)


def _qnn_eltwise_relu_pattern():
    pattern_op = is_op("qnn.add") | is_op("qnn.subtract") | is_op("qnn.mul")
    pattern = pattern_op(
        wildcard(),
        wildcard(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )
    pattern = is_op("clip")(pattern)

    @_checker
    def check(call: relay.Call):
        clip = call
        qnn_eltwise = clip.args[0]
        if get_activation_str(qnn_eltwise.args[6], qnn_eltwise.args[7], clip) is None:
            return False
        return True

    return ("aipu_compass.QnnEltwiseRelu", pattern, check)


@_checker
def _check_dense(dense: relay.Call):
    in_shape = dense.args[0].checked_type.shape
    res = []
    # support 2 dim
    res.append(_check_dim([in_shape], [2]))
    return all(res)


def _dense_pattern():
    pattern = is_op("nn.dense")(wildcard(), is_constant())

    return ("aipu_compass.Dense", pattern, _check_dense)


def _dense_add_pattern():
    pattern = is_op("nn.dense")(wildcard(), is_constant())
    pattern = is_op("add")(pattern, is_constant())

    @_checker
    def check(dense_add: relay.Call):
        # Check if the given match is supported by AIPU Compass.
        dense, bias = unpack_commutative_args(dense_add)

        res = []
        bias_shape = bias.checked_type.shape
        res.append(len(bias_shape) == 1 or bias_shape[0] == 1)
        res.append(_check_dense(dense))

        return all(res)

    return ("aipu_compass.DenseAdd", pattern, check)


def _dense_add_activation_pattern():
    dense = is_op("nn.dense")(wildcard(), is_constant())
    add = is_op("add")(dense, is_constant())
    relu = is_op("nn.relu")(add)
    clip = is_op("clip")(add)
    leaky_relu = is_op("nn.leaky_relu")(add)
    pattern = relu | clip | leaky_relu

    @_checker
    def check(dense_add_activation: relay.Call):
        # Check if the given match is supported by AIPU Compass.
        dense, bias = unpack_commutative_args(dense_add_activation.args[0])

        res = []
        bias_shape = bias.checked_type.shape
        res.append(len(bias_shape) == 1 or bias_shape[0] == 1)
        res.append(_check_dense(dense))

        return all(res)

    return ("aipu_compass.DenseAddActivation", pattern, check)


def _layernorm_pattern0():
    layernorm_input = wildcard()
    mean = is_op("mean")(layernorm_input)
    input_mean = layernorm_input - mean
    sqrt = input_mean * input_mean
    sqrt_mean = is_op("mean")(sqrt)
    epsilon = is_constant()
    rsqrt = input_mean / is_op("power")((sqrt_mean + epsilon), is_constant())
    alpha_mul = rsqrt * is_constant()
    pattern = is_constant() + alpha_mul

    @_checker
    def check(layernorm: relay.Call):
        # Check if the given match is supported by AIPU Compass.
        beta_add = layernorm
        alpha_mul, _ = unpack_commutative_args(beta_add)
        rsqrt, _ = unpack_commutative_args(alpha_mul)
        input_mean, _ = rsqrt.args
        _, input_mean = input_mean.args
        mean_axis0 = input_mean.attrs.axis
        mean_axis0 = [int(axis) for axis in mean_axis0]
        power_op = rsqrt.args[1]
        power_num = power_op.args[1]
        input_mean_mean, epsilon = unpack_commutative_args(power_op.args[0])
        mean_axis1 = input_mean_mean.attrs.axis
        mean_axis1 = [int(axis) for axis in mean_axis1]
        if mean_axis0 != mean_axis1:
            return False
        if not _is_scalar_and_close(power_num, 0.5):
            return False
        if epsilon.data.numpy() > 1e-4:
            return False
        out_shape = beta_add.checked_type.shape
        dim = len(out_shape)
        if len(mean_axis0) != 1 or (mean_axis0[0] != -1 and mean_axis0[0] != dim - 1):
            return False
        yield  # The Compass OP Spec check code must be placed after this statement.
        return True

    return ("aipu_compass.LayerNorm0", pattern, check)


def _layernorm_pattern1():
    inp = wildcard()
    mean = is_op("mean")(inp)
    input_mean = inp - mean
    power = is_op("power")(input_mean, is_constant())
    power_mean = is_op("mean")(power)
    rsqrt = power_mean + is_constant()
    rsqrt = is_op("rsqrt")(rsqrt)
    mul_rsqrt = rsqrt * input_mean
    pattern = mul_rsqrt * is_constant() + is_constant()

    @_checker
    def check(layernorm: relay.Call):
        # Check if the given match is supported by AIPU Compass.
        beta_add = layernorm
        alpha_mul, _ = unpack_commutative_args(beta_add)
        rsqrt_mul, _ = unpack_commutative_args(alpha_mul)
        rsqrt, input_mean = unpack_commutative_args(rsqrt_mul, "subtract")
        rsqrt_inp = rsqrt.args[0]
        power_mean, epsilon = unpack_commutative_args(rsqrt_inp)
        power = power_mean.args[0]
        power_mean_axis = power_mean.attrs.axis
        power_mean_axis = [int(axis) for axis in power_mean_axis]
        power_num = power.args[1]
        mean = input_mean.args[1]
        mean_axis = mean.attrs.axis
        mean_axis = [int(axis) for axis in mean_axis]

        if mean_axis != power_mean_axis:
            return False
        if not _is_scalar_and_close(power_num, 2):
            return False
        if epsilon.data.numpy() > 1e-4:
            return False
        in_shape = mean.args[0].checked_type.shape
        dim = len(in_shape)
        if len(mean_axis) != 1 or (mean_axis[0] != -1 and mean_axis[0] != dim - 1):
            return False

        yield  # The Compass OP Spec check code must be placed after this statement.

        res = []
        res.append(_check_dim([in_shape], [2, 3, 4, 5, 6]))
        return all(res)

    return ("aipu_compass.LayerNorm1", pattern, check)


def _mean_variance_norm_pattern():
    norm_input = wildcard()
    mean = is_op("mean")(norm_input)
    variance = is_op("variance")(norm_input, mean)
    epsilon_add = variance + is_constant()
    mean_delta = norm_input - mean
    divide_sqrt = mean_delta / is_op("sqrt")(epsilon_add)
    multiply_rsqrt = mean_delta * is_op("rsqrt")(epsilon_add)
    pattern = divide_sqrt | multiply_rsqrt

    @_checker
    def check(norm: relay.Call):
        # Check if the given match is supported by AIPU Compass.
        if norm.op == relay.op.get("multiply"):
            _, rsqrt = unpack_commutative_args(norm, "rsqrt")
            epsilon_add = rsqrt.args[0]
        else:
            epsilon_add = norm.args[1].args[0]
        variance, epsilon = unpack_commutative_args(epsilon_add)
        variance_axis = variance.attrs.axis
        variance_axis = [int(val) for val in variance_axis]
        variance_keep_dims = variance.attrs.keepdims
        mean_axis = variance.args[1].attrs.axis
        mean_axis = [int(val) for val in mean_axis]
        if not variance_keep_dims:
            return False
        if mean_axis != variance_axis:
            return False
        if epsilon.data.numpy() > 1e-4:
            return False
        in_shape = variance.args[0].checked_type.shape
        if len(mean_axis) != 1 or mean_axis[0] not in [-1, len(in_shape) - 1]:
            return False

        yield  # The Compass OP Spec check code must be placed after this statement.

        res = []
        # Limited by the Spec of "transpose".
        res.append(_check_dim([in_shape], [1, 2, 3, 4, 5, 6]))
        return all(res)

    return ("aipu_compass.MeanVarianceNormalization", pattern, check)


def _batchnorm_pattern():
    multiply = is_op("multiply")(wildcard(), is_constant())
    batchnorm = is_op("add")(multiply, is_constant())

    @_checker
    def check(batchnorm: relay.Call):
        # Check if the given match is supported by AIPU Compass.
        mul_in, const_in = unpack_commutative_args(batchnorm)
        shape = [int(val) for val in const_in.checked_type.shape]
        if len(shape) != 1 and not all([val == 1 for val in shape[:-1]]):
            return False

        _, const_in = unpack_commutative_args(mul_in)
        shape = [int(val) for val in const_in.checked_type.shape]
        if len(shape) != 1 and not all([val == 1 for val in shape[:-1]]):
            return False

        yield  # The Compass OP Spec check code must be placed after this statement.

        return True

    return ("aipu_compass.BatchNorm", batchnorm, check)


def _batchnorm_single_pattern():
    add = is_op("add")(wildcard(), is_constant())
    multiply = is_op("multiply")(wildcard(), is_constant())
    sub = is_op("subtract")(wildcard(), is_constant())
    batch_norm = add | multiply | sub

    @_checker
    def check(batch_norm: relay.Call):
        # Check if the given match is supported by AIPU Compass.
        arg_in, const_in = unpack_commutative_args(batch_norm)
        if isinstance(arg_in, relay.Constant):
            return False
        if len(arg_in.checked_type.shape) != 4:
            return False
        shape = [int(val) for val in const_in.checked_type.shape]
        if len(shape) != 1 and not all([val == 1 for val in shape[:-1]]):
            return False

        yield  # The Compass OP Spec check code must be placed after this statement.
        # for quantinize convinent, also bn is faster than add/mul/sub
        return True

    return ("aipu_compass.BatchNorm", batch_norm, check)


def _instancenorm_pattern():
    inp = wildcard()
    mean = is_op("mean")(inp)
    variance = is_op("variance")(inp, mean)
    add_epsilon = variance + is_constant()
    rsqrt = is_op("rsqrt")(add_epsilon)
    mean_delta = inp - mean
    mul_rsqrt = mean_delta * rsqrt
    mul_gamma = mul_rsqrt * is_constant()
    instancenorm = mul_gamma + is_constant()

    @_checker
    def check(norm: relay.Call):
        mul_gamma, _ = unpack_commutative_args(norm)
        mul_rsqrt, _ = unpack_commutative_args(mul_gamma)
        mean_delta, _ = unpack_commutative_args(mul_rsqrt, "rsqrt")
        inp = mean_delta.args[0]

        in_shape = inp.checked_type.shape
        out_shape = norm.checked_type.shape

        res = []
        # only support 4-dim
        res.append(_check_dim([in_shape, out_shape], [4]))
        # check shape of input and output
        res.append(_check_range(in_shape[1:], [16384, 16384, 16384]))
        res.append(_check_range(out_shape[1:], [16384, 16384, 16384]))
        return all(res)

    return ("aipu_compass.InstanceNorm", instancenorm, check)


@ir.register_op_attr("copy", "target.aipu_compass")
@ir.register_op_attr("transpose", "target.aipu_compass")
@ir.register_op_attr("transpose", "target.aipu_compass.qnn")
@_checker
def _check_transpose(transpose: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape = transpose.args[0].checked_type.shape
    res = []
    # support 1, 2, 3, 4, 5, 6 dim
    res.append(_check_dim([in_shape], [1, 2, 3, 4, 5, 6]))
    return all(res)


@ir.register_op_attr("contrib.aipu_compass.channel_shuffle", "target.aipu_compass")
@_checker
def _check(channel_shuffle: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape = channel_shuffle.args[0].checked_type.shape
    out_shape = channel_shuffle.checked_type.shape
    group = channel_shuffle.attrs.group

    res = []
    # support 2, 3, 4 dim
    res.append(_check_dim([in_shape, out_shape], [2, 3, 4]))
    # check shape of input and output
    res.append(_check_shape(in_shape, 0, 0, 4))
    res.append(_check_shape(out_shape, 0, 0, 4))
    # check group
    res.append(1 <= group <= 16384)

    return all(res)


def peel_hardswish(hardswish):
    """Peel the needed nodes out."""
    args = hardswish.args
    if any(isinstance(arg, relay.Constant) for arg in args):
        # The last OP is multiply 1/6 or divide 6.
        # (clip(x + 3) * x) / 6 or (clip(x + 3) * x) * 1/6
        six_mul_or_div = hardswish
        add_clip_mul, _ = unpack_commutative_args(six_mul_or_div)
        data, clip = unpack_commutative_args(add_clip_mul, "clip")
    else:
        # The last OP is the other multiply.
        if any(isinstance(arg, relay.Call) and arg.op == relay.op.get("clip") for arg in args):
            # x / 6 * clip(x + 3) or x * 1/6 * clip(x + 3)
            six_mul_or_div, clip = unpack_commutative_args(hardswish, "clip")
        else:
            # clip(x + 3) / 6 * x or clip(x + 3) * 1/6 * x
            six_mul_or_div = [
                arg
                for arg in args
                if isinstance(arg, relay.Call)
                and arg.op in (relay.op.get("multiply"), relay.op.get("divide"))
            ]

            def _check_six(call):
                if all([not isinstance(arg, relay.Constant) for arg in call.args]):
                    return False
                _, const_v = unpack_commutative_args(call)
                if call.op == relay.op.get("multiply") and _is_scalar_and_close(const_v, 1 / 6):
                    return True
                if call.op == relay.op.get("divide") and _is_scalar_and_close(const_v, 6):
                    return True
                return False

            six_mul_or_div = [expr for expr in six_mul_or_div if _check_six(expr)][0]
            clip, _ = unpack_commutative_args(six_mul_or_div)
        data, _ = unpack_commutative_args(clip.args[0])
    return data, six_mul_or_div, clip


@_checker
def _check_hardswish(hardswish: relay.Call):
    data, six_mul_or_div, clip = peel_hardswish(hardswish)
    _, six_const = unpack_commutative_args(six_mul_or_div)
    ref = 1 / 6 if six_mul_or_div.op == relay.op.get("multiply") else 6
    if not _is_scalar_and_close(six_const, ref):
        return False

    _, add_const = unpack_commutative_args(clip.args[0])
    if not _is_scalar_and_close(add_const, 3):
        return False

    yield  # The Compass OP Spec check code must be placed after this statement.

    in_shape = data.checked_type.shape
    res = []
    res.append(_check_dim([in_shape], [2, 3, 4]))
    res.append(_check_shape(in_shape, 0, 0, 4))
    return all(res)


def _hardswish_pattern():
    data = wildcard()
    add = is_op("add")(data, is_constant())
    add_clip = is_op("clip")(add).has_attr({"a_min": 0.0, "a_max": 6.0})
    add_clip_mul = is_op("multiply")(data, add_clip)
    add_clip_mul_mul = is_op("multiply")(add_clip_mul, is_constant())
    add_clip_mul_div = is_op("divide")(add_clip_mul, is_constant())

    pattern = add_clip_mul_mul | add_clip_mul_div

    return ("aipu_compass.HardSwish", pattern, _check_hardswish)


def _qnn_hardswish_pattern():
    dequant = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
    dequant_add = is_op("add")(dequant, is_constant())
    dequant_add_clip = is_op("clip")(dequant_add)
    dequant_add_clip_mul = is_op("multiply")(dequant, dequant_add_clip)
    dequant_add_clip_mul_mul = is_op("multiply")(dequant_add_clip_mul, is_constant())
    dequant_add_clip_mul_mul_quant = is_op("qnn.quantize")(
        dequant_add_clip_mul_mul, is_constant(), is_constant()
    )
    dequant_add_clip_mul_div = is_op("divide")(dequant_add_clip_mul, is_constant())
    dequant_add_clip_mul_div_quant = is_op("qnn.quantize")(
        dequant_add_clip_mul_div, is_constant(), is_constant()
    )

    pattern = dequant_add_clip_mul_mul_quant | dequant_add_clip_mul_div_quant

    return ("aipu_compass.QnnHardSwish", pattern, lambda x: _check_hardswish(x.args[0]))


@ir.register_op_attr("nn.log_softmax", "target.aipu_compass")
@ir.register_op_attr("nn.softmax", "target.aipu_compass")
@_checker
def _check_softmax(softmax: relay.Call):
    in_shape = softmax.args[0].checked_type.shape
    res = []
    res.append(_check_transpose(softmax))
    res.append(-1 <= int(softmax.attrs.axis) <= len(in_shape) - 1)

    return all(res)


def _qnn_softmax_pattern():
    dequantize = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
    dequantize_softmax = is_op("nn.softmax")(dequantize)
    dequantize_softmax_quantize = is_op("qnn.quantize")(
        dequantize_softmax, is_constant(), is_constant()
    )

    pattern = dequantize_softmax_quantize

    return ("aipu_compass.QnnSoftmax", pattern, lambda x: _check_softmax(x.args[0]))


def _qnn_resize2d_pattern():
    dequantize = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
    dequantize_resize2d = is_op("image.resize2d")(dequantize)
    dequantize_resize2d_quantize = is_op("qnn.quantize")(
        dequantize_resize2d, is_constant(), is_constant()
    )

    pattern = dequantize_resize2d_quantize

    @_checker
    def check(call: relay.Call):
        # Check if the given match is supported by AIPU Compass.
        resize2d = call.args[0]
        return _resize2d_check(resize2d)

    return ("aipu_compass.QnnResize2D", pattern, check)


def _qnn_flatten_prelu_pattern():
    reshape0 = is_op("reshape")(wildcard())
    prelu = is_op("qnn.prelu")(
        reshape0,
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )
    reshape1 = is_op("reshape")(prelu)
    pattern = reshape1

    return ("aipu_compass.QnnFlattenPrelu", pattern, lambda x: _check_prelu(x.args[0]))


def _qnn_squared_difference_pattern():
    difference = is_op("qnn.subtract")(
        wildcard(),
        wildcard(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )
    squared_difference = is_op("qnn.mul")(
        difference,
        difference,
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )

    return (
        "aipu_compass.QnnSquaredDifference",
        squared_difference,
        lambda x: _check_nothing(x.args[0]),
    )


def _log_softmax_pattern():
    pattern = is_op("log")(is_op("nn.softmax")(wildcard()))

    return ("aipu_compass.LogSoftmax", pattern, lambda x: _check_softmax(x.args[0]))


def _qnn_requant_s2d_pattern():
    requantize = is_op("qnn.requantize")(
        is_var(), is_constant(), is_constant(), is_constant(), is_constant()
    )
    pattern = is_op("nn.space_to_depth")(requantize)

    return ("aipu_compass.QnnRequantS2D", pattern, _check_space_to_depth)


@ir.register_op_attr("nn.avg_pool2d", "target.aipu_compass")
@_checker
def _check_avg_pool2d(avg_pool2d: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape = avg_pool2d.args[0].checked_type.shape
    attrs = avg_pool2d.attrs

    res = []
    # only support 4-dim
    res.append(_check_dim([in_shape], [4]))
    # check shape of input and output
    res.append(_check_range(in_shape[:], [32, 16384, 1920, 16384]))
    # check kernel
    kernels = attrs.pool_size
    max_int32 = 2**31 - 1
    input_data_size = 32
    res.append(
        _check_range(kernels, [65, 65])
        or (
            kernels[1] == in_shape[2]
            and kernels[0] == in_shape[1]
            and (
                kernels[1] * kernels[0] < 28 * 1024
                or kernels[1] * kernels[0] * input_data_size < max_int32
            )
        )
    )
    # check stride. (beyond spec)
    strides = attrs.strides
    res.append(
        _check_range(strides, [32, 32]) or (kernels[1] == in_shape[2] and kernels[0] == in_shape[1])
    )
    res.append(
        (strides[0] <= kernels[0] and strides[1] <= kernels[1])
        or (kernels[0] == kernels[1] == 1 and strides[0] == strides[1] == 2)
    )
    # check pad
    pads = attrs.padding
    res.append(_check_range(pads, [6, 6, 6, 6], 0))
    res.append(all(p < kernels[0] for p in pads[::2]) and all(p < kernels[1] for p in pads[1::2]))

    return all(res)


def _qnn_avg_pool2d_pattern():
    cast = is_op("cast")(wildcard()).has_attr({"dtype": "int32"})
    cast_avg_pool2d = is_op("nn.avg_pool2d")(cast)
    cast_avg_pool2d_cast = is_op("cast")(cast_avg_pool2d)

    pattern = cast_avg_pool2d_cast

    return ("aipu_compass.QnnAvgPool2D", pattern, lambda x: _check_avg_pool2d(x.args[0]))


def _qnn_reduce_pattern():
    cast = is_op("cast")(wildcard()).has_attr({"dtype": "int32"})
    cast_reduce = is_op("mean")(cast)
    cast_reduce_requant = is_op("qnn.requantize")(
        cast_reduce, is_constant(), is_constant(), is_constant(), is_constant()
    )

    pattern = cast_reduce_requant

    return ("aipu_compass.QnnReduce", pattern, lambda x: _check_reduce(x.args[0]))


def _qnn_dense_add_pattern():
    qnn_dense = is_op("qnn.dense")(
        wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
    )
    qnn_dense_add = is_op("add")(qnn_dense, is_constant())
    qnn_dense_add_req = is_op("qnn.requantize")(
        qnn_dense_add, is_constant(), is_constant(), is_constant(), is_constant()
    )

    @_checker
    def check(call: relay.Call):
        dense, bias = unpack_commutative_args(call.args[0])

        res = []
        bias_shape = bias.checked_type.shape
        res.append(len(bias_shape) == 1 or bias_shape[0] == 1)
        res.append(_check_dense(dense))

        return all(res)

    return ("aipu_compass.QnnDenseAdd", qnn_dense_add_req, check)


def _qnn_dense_pattern():
    qnn_dense = is_op("qnn.dense")(
        wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
    )
    qnn_dense_requant = is_op("qnn.requantize")(
        qnn_dense, is_constant(), is_constant(), is_constant(), is_constant()
    )
    qdense_req_clip = is_op("clip")(qnn_dense_requant)
    pattern = qnn_dense_requant | qdense_req_clip

    @_checker
    def check(call: relay.Call):
        if qnn_dense_requant.match(call):
            requantize = call
        elif qdense_req_clip.match(call):
            clip = call
            requantize = clip.args[0]
            if get_activation_str(requantize.args[3], requantize.args[4], clip) is None:
                return False
        dense = requantize.args[0]

        return _check_dense(dense)

    return ("aipu_compass.QnnDense", pattern, check)


def _qnn_matmul_pattern():
    reshape0 = is_op("reshape")(wildcard())
    reshape1 = is_op("reshape")(wildcard())
    reshape1_trans = is_op("transpose")(reshape1)
    qdense = is_op("qnn.dense")(
        reshape0, reshape1_trans, is_constant(), is_constant(), is_constant(), is_constant()
    )
    qdense_req = is_op("qnn.requantize")(
        qdense, is_constant(), is_constant(), is_constant(), is_constant()
    )
    qdense_req_reshape = is_op("expand_dims")(qdense_req)
    pattern = qdense_req_reshape

    @_checker
    def check(call: relay.Call):
        expand_dims = call
        requantize = expand_dims.args[0]
        qdense = requantize.args[0]
        reshape0 = qdense.args[0]
        transpose = qdense.args[1]
        reshape1 = transpose.args[0]
        reshape0_shape = reshape0.checked_type.shape
        reshape1_shape = reshape1.checked_type.shape
        trans_axes = [int(x) for x in transpose.attrs.axes]
        res = []
        res.append(len(reshape0_shape) == 2 and len(reshape1_shape) == 2)
        res.append(reshape0_shape[1] == reshape1_shape[0])
        res.append(trans_axes == [1, 0])
        res.append(expand_dims.attrs.axis == 0)

        return all(res)

    return ("aipu_compass.QnnMatmul", pattern, check)


def _qnn_cast_pattern():
    cast = is_op("cast")(wildcard())
    cast_q = is_op("qnn.quantize")(cast, is_constant(), is_constant())
    pattern = cast_q

    return ("aipu_compass.QnnCast", pattern, lambda x: _check_nothing(x.args[0]))


def _qnn_mirror_pad_pattern():
    mirror_pad = is_op("nn.mirror_pad")(wildcard())
    mirror_pad_req = is_op("qnn.requantize")(
        mirror_pad, is_constant(), is_constant(), is_constant(), is_constant()
    )
    pattern = mirror_pad_req

    return ("aipu_compass.QnnMirrorPad", pattern, lambda x: _check_mirror_pad(x.args[0]))


def _qnn_sigmoid_pattern():
    # req + qsigmoid
    qsigmoid = is_op("qnn.sigmoid")(
        wildcard(), is_constant(), is_constant(), is_constant(), is_constant()
    )
    qsigmoid_req = is_op("qnn.requantize")(
        qsigmoid, is_constant(), is_constant(), is_constant(), is_constant()
    )
    # deq + sigmoid + q
    dequantize = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
    dequantize_sigmoid = is_op("sigmoid")(dequantize)
    dequantize_sigmoid_quantize = is_op("qnn.quantize")(
        dequantize_sigmoid, is_constant(), is_constant()
    )

    pattern = qsigmoid_req | dequantize_sigmoid_quantize
    return ("aipu_compass.QnnSigmoid", pattern, lambda x: _check_nothing(x.args[0]))


def _qnn_silu_pattern():
    inp = wildcard()
    sigmoid = is_op("qnn.sigmoid")(inp, is_constant(), is_constant(), is_constant(), is_constant())
    mult = is_op("qnn.mul")(
        inp,
        sigmoid,
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )

    return ("aipu_compass.QnnSilu", mult, _check_nothing)


def _qnn_minimum_pattern():
    inp = wildcard()
    qmin = is_op("minimum")(inp, wildcard())
    qmin_clip = is_op("clip")(qmin)
    qmin_clip_req = is_op("qnn.requantize")(
        qmin_clip, is_constant(), is_constant(), is_constant(), is_constant()
    )
    pattern = qmin_clip_req

    @_checker
    def check(call: relay.Call):
        requantize = call
        clip = requantize.args[0]
        if get_activation_str(requantize.args[1], requantize.args[2], clip) is None:
            return False
        return True

    return ("aipu_compass.QnnMinimum", pattern, check)


def pattern_table_pre(include_float, include_quant):
    """
    The AIPU Compass pattern table.
    Should be composited before aipu op convert
    """
    float_patterns = [
        _hardswish_pattern(),
        _convolution2d_pattern(),
        _convolution3d_pattern(),
        _dense_add_activation_pattern(),
        _dense_add_pattern(),
        _dense_pattern(),
        _layernorm_pattern0(),
        _layernorm_pattern1(),
        _softplus_pattern(),
        _instancenorm_pattern(),
        _batchnorm_pattern(),
        _mean_variance_norm_pattern(),
        _log_softmax_pattern(),
        _batchnorm_single_pattern(),
    ]
    quant_patterns = [
        _qnn_basiclstm_pattern(),
        _qnn_convolution2d_pattern(),
        _qnn_softmax_pattern(),
        _qnn_avg_pool2d_pattern(),
        _qnn_hardswish_pattern(),
        _qnn_reduce_pattern(),
        _qnn_requant_s2d_pattern(),
        _qnn_matmul_pattern(),
        _qnn_dense_add_pattern(),
        _qnn_dense_pattern(),
        _qnn_silu_pattern(),
        _qnn_flatten_prelu_pattern(),
        _qnn_cast_pattern(),
        _qnn_sigmoid_pattern(),
        _qnn_mirror_pad_pattern(),
        _qnn_resize2d_pattern(),
    ]
    ret = []
    if include_float:
        ret += float_patterns
    if include_quant:
        ret += quant_patterns
    return ret


def pattern_table_post(include_float, include_quant):
    """
    The AIPU Compass pattern table.
    Composited after aipu op convert
    """
    float_patterns = [
        _elementwise_relu_pattern(),
    ]
    quant_patterns = [
        _qnn_eltwise_relu_pattern(),
        _qnn_squared_difference_pattern(),
        _qnn_avg_pool2d_pattern(),
        _qnn_minimum_pattern(),
    ]
    ret = []
    if include_float:
        ret += float_patterns
    if include_quant:
        ret += quant_patterns
    return ret


@ir.register_op_attr("nn.max_pool2d", "target.aipu_compass.qnn")
@ir.register_op_attr("nn.max_pool2d", "target.aipu_compass")
@_checker
def _check(max_pool2d: relay.Call):
    in_shape = max_pool2d.args[0].checked_type.shape
    attrs = max_pool2d.attrs

    res = []
    # only support 4-dim
    res.append(_check_dim([in_shape], [4]))
    # check shape of input
    res.append(_check_range(in_shape[:], [32, 16384, 1920, 16384]))
    # check kernel
    kernels = attrs.pool_size
    res.append(_check_range(kernels, [65, 65]))
    # check stride
    strides = attrs.strides
    res.append(_check_range(strides, [16, 16]))
    res.append(
        (strides[0] <= kernels[0] and strides[1] <= kernels[1])
        or (kernels[0] == kernels[1] == 1 and strides[0] == strides[1] == 2)
    )
    # check pad
    pads = attrs.padding
    res.append(_check_range(pads, [16, 16, 16, 16], 0))
    res.append(all(p < kernels[0] for p in pads[::2]) and all(p < kernels[1] for p in pads[1::2]))

    return all(res)


@ir.register_op_attr("nn.global_avg_pool2d", "target.aipu_compass")
@_checker
def _check(global_avg_pool2d: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape = global_avg_pool2d.args[0].checked_type.shape
    out_shape = global_avg_pool2d.checked_type.shape

    res = []
    # only support 4-dim
    res.append(_check_dim([in_shape, out_shape], [4]))
    # check shape of input and output
    res.append(_check_range(in_shape[:], [32, 16384, 1920, 16384]))
    res.append(_check_range(out_shape[:], [32, 16384, 1920, 16384]))
    # check kernel
    kernel_x = in_shape[2]
    kernel_y = in_shape[1]
    max_int32 = 2**31 - 1
    input_data_size = 32
    res.append(
        _check_range(in_shape[1:3], [65, 65])
        or kernel_x * kernel_y < 28 * 1024
        or kernel_x * kernel_y * input_data_size < max_int32
    )

    return all(res)


@ir.register_op_attr("nn.global_max_pool2d", "target.aipu_compass")
@_checker
def _check(global_max_pool2d: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape = global_max_pool2d.args[0].checked_type.shape

    res = []
    # only support 4-dim
    res.append(_check_dim([in_shape], [4]))
    # check shape of input
    res.append(_check_range(in_shape[:], [32, 16384, 1920, 16384]))
    # check kernel
    res.append(_check_range(in_shape[1:3], [65, 65]))

    return all(res)


@ir.register_op_attr("concatenate", "target.aipu_compass.qnn")
@ir.register_op_attr("qnn.concatenate", "target.aipu_compass.qnn")
@ir.register_op_attr("concatenate", "target.aipu_compass")
@_checker
def _check(concatenate: relay.Call):
    # Check if it is supported by AIPU Compass.
    res = []
    res.append(len(concatenate.args[0]) >= 2)

    return all(res)


@ir.register_op_attr("mod", "target.aipu_compass")
@_checker
def _check(mod: relay.Call):
    # Check if it is supported by AIPU Compass.
    in0_shape = mod.args[0].checked_type.shape
    in1_shape = mod.args[1].checked_type.shape

    res = []
    # support 2, 3, 4 dim
    res.append(_check_dim([in0_shape, in1_shape], [2, 3, 4]))
    # check shape of input and output
    res.append(_check_shape(in0_shape, 0, 0, 4))
    res.append(_check_shape(in1_shape, 0, 0, 4))

    return all(res)


@ir.register_op_attr("one_hot", "target.aipu_compass")
@_checker
def _check(one_hot: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape = one_hot.args[0].checked_type.shape
    on_value = one_hot.args[1]
    off_value = one_hot.args[2]
    attrs = one_hot.attrs
    dim = len(in_shape)

    if not isinstance(on_value, relay.Constant) or not isinstance(off_value, relay.Constant):
        return False

    yield  # The Compass OP Spec check code must be placed after this statement.

    on_value = on_value.data.numpy().item()
    off_value = off_value.data.numpy().item()

    res = []
    # support 2, 3, 4, 5 dim
    res.append(_check_dim([in_shape], [2, 3, 4, 5], False))
    spec = [32] + [16384, 16384, 16384][4 - dim :]
    res.append(_check_range(in_shape, spec))
    # check depth and on/off value
    res.append(_check_range([attrs.depth], [16384]))
    res.append(_check_range([on_value, off_value], [65535, 65535], 0))
    # check axis
    res.append(-1 <= attrs.axis <= 4)

    return all(res)


@ir.register_op_attr("image.grid_sample", "target.aipu_compass")
@_checker
def _check(grid_sample: relay.Call):
    # Check if it is supported by AIPU Compass.
    data_shape = grid_sample.args[0].checked_type.shape
    grid_shape = grid_sample.args[1].checked_type.shape
    attrs = grid_sample.attrs

    res = []
    # only support 4-dim
    res.append(_check_dim([data_shape, grid_shape], [4]))
    # check shape of input
    res.append(_check_shape(data_shape, 0, 0, 4))
    res.append(_check_range(grid_shape[1:-1], [16384, 16384]))
    res.append(grid_shape[-1] == 2)
    # check method
    res.append(attrs.method in ["nearest", "bilinear"])
    # check padding_mode
    res.append(attrs.padding_mode in ["zeros", "border"])

    return all(res)


@ir.register_op_attr("split", "target.aipu_compass.qnn")
@ir.register_op_attr("split", "target.aipu_compass")
@_checker
def _check(split: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape = split.args[0].checked_type.shape
    out_shapes = [tt.shape for tt in split.checked_type.fields]

    res = []
    # check output tensor number
    res.append(len(out_shapes) >= 2)
    # support 2, 3, 4, 5 dim
    res.append(_check_dim([in_shape] + out_shapes, [2, 3, 4, 5]))
    dim = len(in_shape)
    res.append(_check_range(in_shape[:], [128] + [16384] * (dim - 1)))
    return all(res)


@ir.register_op_attr("nn.pad", "target.aipu_compass.qnn")
@ir.register_op_attr("nn.pad", "target.aipu_compass")
@_checker
def _check(pad: relay.Call):
    # Check if it is supported by AIPU Compass.
    # check pad_value
    pad_value = pad.args[1]
    if not isinstance(pad_value, relay.Constant):
        return False
    # Check pad_mod.
    attrs = pad.attrs
    if attrs.pad_mode not in ["constant", "reflect"]:
        return False

    yield  # The Compass OP Spec check code must be placed after this statement.

    in_shape = pad.args[0].checked_type.shape
    out_shape = pad.checked_type.shape
    res = []
    # support 2, 3, 4, 5 dim
    dim = len(in_shape)
    res.append(_check_dim([in_shape, out_shape], [2, 3, 4, 5]))
    res.append(_check_range(in_shape[1:], [16384] * (dim - 1)))

    # check pad_width
    pad_width = attrs.pad_width
    for i in range(len(in_shape)):
        res.append(_check_range(pad_width[i], [128, 128], -1))

    return all(res)


@ir.register_op_attr("nn.prelu", "target.aipu_compass")
@_checker
def _check_prelu(prelu: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape = prelu.args[0].checked_type.shape
    # Check axis
    dim = len(in_shape)
    axis = prelu.attrs.axis
    if axis not in (dim - 1, -1):
        return False
    return True


@ir.register_op_attr("nn.lrn", "target.aipu_compass")
@_checker
def _check(lrn: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape = lrn.args[0].checked_type.shape
    out_shape = lrn.checked_type.shape
    attrs = lrn.attrs

    res = []
    # only support 4-dim
    res.append(_check_dim([in_shape, out_shape], [4]))
    # check shape of input and output
    res.append(_check_shape(in_shape, 0, 0, 4))
    # check size
    res.append(1 <= attrs.size <= 64)
    # check bias
    res.append(0 < attrs.bias <= 64)
    # check alpha
    res.append(0 < attrs.alpha <= 64)
    # check beta
    res.append(0 < attrs.beta < 1)
    # check axis which not support N
    res.append(attrs.axis != 0)

    return all(res)


@ir.register_op_attr("argmin", "target.aipu_compass")
@ir.register_op_attr("argmax", "target.aipu_compass")
@ir.register_op_attr("argmin", "target.aipu_compass.qnn")
@ir.register_op_attr("argmax", "target.aipu_compass.qnn")
@_checker
def _check(argmax: relay.Call):
    # Check if it is supported by AIPU Compass.
    attrs = argmax.attrs
    if not attrs.keepdims or attrs.exclude or len(attrs.axis) != 1:
        return False

    yield  # The Compass OP Spec check code must be placed after this statement.

    in_shape = argmax.args[0].checked_type.shape
    res = []
    res.append(_check_dim([in_shape], [2, 3, 4, 5]))
    return all(res)


@ir.register_op_attr("reverse_sequence", "target.aipu_compass.qnn")
@ir.register_op_attr("reverse_sequence", "target.aipu_compass")
@_checker
def _check(reverse_sequence: relay.Call):
    attrs = reverse_sequence.attrs
    # Check if it is supported by AIPU Compass.
    in_shape = reverse_sequence.args[0].checked_type.shape
    out_shape = reverse_sequence.checked_type.shape

    res = []
    # support 2, 3 dim
    res.append(_check_dim([in_shape, out_shape], [2, 3]))
    # check shape of input and output
    res.append(_check_shape(in_shape, 0, 0, 3))
    batch_axis = int(attrs.batch_axis)
    seq_axis = int(attrs.seq_axis)
    # res.append(0 <= batch_axis <= 1)
    res.append(0 <= seq_axis <= 1 or (seq_axis == 2 and len(in_shape) == 3))
    res.append(batch_axis != seq_axis)

    return all(res)


@ir.register_op_attr("nn.batch_matmul", "target.aipu_compass")
@_checker
def _check(batch_matmul: relay.Call):
    in_shape1 = batch_matmul.args[0].checked_type.shape
    in_shape2 = batch_matmul.args[1].checked_type.shape

    res = []
    # check shape of input and output
    res.append(_check_range(in_shape1[1:], [16384] * 2))
    res.append(_check_range(in_shape2[1:], [16384] * 2))

    return all(res)


@ir.register_op_attr("contrib.aipu_compass.gruv3", "target.aipu_compass")
@_checker
def _check(gruv3: relay.Call):
    # Check if it is supported by AIPU Compass.
    attrs = gruv3.attrs
    in_shape0 = gruv3.args[0].checked_type.shape
    in_shape1 = gruv3.args[1].checked_type.shape
    in_type1 = gruv3.args[1].checked_type.dtype
    out_ttype = gruv3.checked_type
    if isinstance(out_ttype, ir.TupleType):
        out_ttype = out_ttype.fields[0]
    res = []
    res.append(len(in_shape0) == 3 and _check_range(in_shape0[:], [16384, 4096, 16384]))
    res.append(len(in_shape1) == 2 and _check_range(in_shape1[:], [16384, 8192]))
    res.append(out_ttype.dtype == in_type1)
    res.append(attrs.out_sequence in ["H", "Hn", "H, Hn"])
    res.append(
        all(
            x.strip().lower()
            in [
                "relu",
                "tanh",
                "sigmoid",
                "affine",
                "leakyrelu",
                "thresholdedrelu",
                "hardsighmoid",
                "elu",
                "softsign",
                "softplus",
            ]
            for x in attrs.activations.split(",")
        )
    )
    return all(res)


@ir.register_op_attr("take", "target.aipu_compass.qnn")
@ir.register_op_attr("take", "target.aipu_compass")
@_checker
def _check(take: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape0 = take.args[0].checked_type.shape
    res = []
    axis = int(take.attrs.axis)
    if axis < 0:
        axis = len(in_shape0) - axis
    res.append(0 <= int(take.attrs.batch_dims) <= axis)

    return all(res)


@ir.register_op_attr("contrib.aipu_compass.ctc_greedy_decoder", "target.aipu_compass")
@_checker
def _check(ctc: relay.Call):
    # Check if it is supported by AIPU Compass.
    data_shape = ctc.args[0].checked_type.shape
    seq_shape = ctc.args[1].checked_type.shape
    out_shape = ctc.checked_type.shape

    res = []
    # check dim
    res.append(_check_dim([data_shape], [3]))
    res.append(_check_dim([seq_shape], [1]))
    res.append(_check_dim([out_shape], [4]))
    # check shape of input and output
    res.append(_check_range(data_shape, [32, 16384, 16384]))
    res.append(_check_range(seq_shape, [32]))
    res.append(1 <= out_shape[0] <= 32)
    res.append(out_shape[1] == 4096 and out_shape[2] == out_shape[3] == 1)
    res.append(bool(ctc.attrs.merge_repeated))
    return all(res)


@ir.register_op_attr("round", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape = call.args[0].checked_type.shape

    res = []
    # support 2, 3, 4, 5 dim
    res.append(_check_dim([in_shape], [2, 3, 4, 5]))
    # check shape of input
    res.append(_check_shape(in_shape, 0, 0, 5))

    return all(res)


@ir.register_op_attr("nn.mirror_pad", "target.aipu_compass.qnn")
@ir.register_op_attr("nn.mirror_pad", "target.aipu_compass")
@_checker
def _check_mirror_pad(pad: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape = pad.args[0].checked_type.shape
    out_shape = pad.checked_type.shape
    attrs = pad.attrs

    res = []
    # support 2, 3, 4, 5 dim
    res.append(_check_dim([in_shape, out_shape], [2, 3, 4, 5]))
    # check shape of input and output
    # for different dims have different spec
    dim = len(out_shape)
    res.append(
        (dim == 2 and _check_range(in_shape[:], [32, 4096]))
        or (dim == 3 and _check_range(in_shape[:], [32, 1920, 4096]))
        or (dim == 4 and _check_range(in_shape[:], [32, 16384, 16384, 16384]))
        or (dim == 5 and _check_range(in_shape[:], [32, 100, 1080, 1920, 4096]))
    )
    # check pad_width
    pad_width = attrs.pad_width
    for i in range(dim):
        res.append(_check_range(pad_width[i], [16, 16], 0))

    return all(res)


@ir.register_op_attr("scatter_nd", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    if not isinstance(call.args[1], relay.Constant):
        return False

    yield  # The Compass OP Spec check code must be placed after this statement.

    in_shape0 = call.args[0].checked_type.shape
    in_shape1 = call.args[1].checked_type.shape
    in_shape2 = call.args[2].checked_type.shape

    res = []
    res.append(_check_dim([in_shape0], [1, 2, 3, 4, 5]))

    res.append(
        in_shape1[0] <= len(in_shape0)
        and all(
            [
                i == j
                for i, j in zip(
                    in_shape2,
                    in_shape1[1:] + in_shape0[int(in_shape1[0]) :],
                )
            ]
        )
    )

    return all(res)


@ir.register_op_attr("left_shift", "target.aipu_compass")
@ir.register_op_attr("right_shift", "target.aipu_compass")
@_checker
def _check(right_shift: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape0 = _convert_array(right_shift.args[0].checked_type.shape)
    in_shape1 = _convert_array(right_shift.args[1].checked_type.shape)

    res = []
    # support 2, 3, 4 dim
    res.append(_check_dim([in_shape0, in_shape1], [2, 3, 4]))
    # Broadcast is not supported, and both inputs should share the same shape
    res.append(len(in_shape0) == len(in_shape1))
    res.append(all([i == j for i, j in zip(in_shape0, in_shape1)]))
    res.append(_check_shape(in_shape0, 0, 1, 4))
    res.append(_check_shape(in_shape1, 0, 1, 4))
    return all(res)


@ir.register_op_attr("sign", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape0 = call.args[0].checked_type.shape

    res = []
    res.append(_check_dim([in_shape0], [2, 3, 4]))
    res.append(_check_shape(in_shape0, 0, 0, 4))

    return all(res)


@ir.register_op_attr("topk", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape0 = call.args[0].checked_type.shape
    in_shape0 = _convert_array(in_shape0)

    attrs = call.attrs

    res = []
    res.append(_check_dim([in_shape0], [1, 2, 3, 4, 5]))
    res.append(
        (len(in_shape0) == 1 and _check_range(in_shape0[:], [16384]))
        or (len(in_shape0) > 1 and _check_shape(in_shape0, 0, 0, 5))
    )
    axis = int(attrs.axis)
    k = int(attrs.k)
    k = in_shape0[axis + len(in_shape0) if axis == -1 else axis] if k < 1 else k
    res.append(1 <= k <= 16384)
    res.append(attrs.ret_type == "both")
    val = in_shape0[axis]
    res.append(val < (1 << 16))
    return all(res)


@ir.register_op_attr("scatter_elements", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape0 = _convert_array(call.args[0].checked_type.shape)
    in_shape1 = _convert_array(call.args[1].checked_type.shape)
    in_shape2 = _convert_array(call.args[2].checked_type.shape)

    attrs = call.attrs

    res = []

    if attrs.reduction == "add":
        res.append(all(x < 16384 for x in in_shape0))
    elif attrs.reduction == "mul":
        res.append(all(x < 4096 for x in in_shape0))
    elif attrs.reduction != "update":
        res.append(False)

    res.append(len(in_shape0) == len(in_shape1))
    res.append(all([i <= j for i, j in zip(in_shape1, in_shape0)]))
    res.append(all([i == j for i, j in zip(in_shape2, in_shape1)]))

    return all(res)


@ir.register_op_attr("vision.roi_align", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape0 = call.args[0].checked_type.shape
    in_shape1 = call.args[1].checked_type.shape

    res = []
    res.append(len(in_shape0) == 4)
    res.append(len(in_shape1) == 2)

    res.append(_check_range(in_shape0[:], [32, 16384, 16384, 16384]))
    res.append(_check_range(in_shape1[1:], [16384]))
    res.append(int(in_shape1[1]) == 5)
    return all(res)


@ir.register_op_attr("vision.roi_pool", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape0 = call.args[0].checked_type.shape
    in_shape1 = call.args[1].checked_type.shape

    res = []
    res.append(len(in_shape0) == 4)
    res.append(len(in_shape1) == 2)

    res.append(_check_range(in_shape0[:], [32, 1080, 1920, 4096]))
    res.append(_check_range(in_shape1[1:], [16384]))
    res.append(int(in_shape1[1]) == 5)
    # opt only support batch == 1, need fix later.
    res.append(in_shape0[0] == 1)
    return all(res)


@ir.register_op_attr("image.crop_and_resize", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape0 = call.args[0].checked_type.shape
    in_shape1 = call.args[1].checked_type.shape
    in_shape2 = call.args[2].checked_type.shape

    res = []
    res.append(len(in_shape0) == 4)
    res.append(len(in_shape1) == 2)
    res.append(len(in_shape2) == 1)

    res.append(_check_shape(in_shape0, 0, 1, 4))
    res.append(_check_range(in_shape1[:1], [16384]))
    res.append(int(in_shape1[1]) == 4)
    res.append(_check_range(in_shape2, [16384]))

    res.append(0 <= int(call.attrs.crop_size[0]) <= 1080)
    res.append(1 <= int(call.attrs.crop_size[1]) <= 1920)

    res.append(call.attrs.method in ("nearest_neighbor", "bilinear"))
    res.append(int(call.attrs.extrapolation_value) in (0, 1))

    return all(res)


@ir.register_op_attr("nn.batch_to_space_nd", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape0 = call.args[0].checked_type.shape
    if len(in_shape0) != 4:
        return False

    yield  # The Compass OP Spec check code must be placed after this statement.

    res = []
    res.append(_check_shape(in_shape0, 0, 1, 4))
    for block_size in call.attrs.block_shape:
        res.append(1 <= int(block_size) <= 16)
    for crop in call.attrs.crops:
        for size in crop:
            res.append(0 <= int(size) <= 16)

    return all(res)


@ir.register_op_attr("nn.space_to_batch_nd", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape0 = call.args[0].checked_type.shape
    if len(in_shape0) != 4:
        return False

    yield  # The Compass OP Spec check code must be placed after this statement.

    res = []
    res.append(_check_shape(in_shape0, 0, 0, 4))
    for block_size in call.attrs.block_shape:
        res.append(1 <= int(block_size) <= 16)
    for pad in call.attrs.paddings:
        for size in pad:
            res.append(0 <= int(size) <= 16)

    return all(res)


@ir.register_op_attr("nn.space_to_depth", "target.aipu_compass")
@_checker
def _check_space_to_depth(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape0 = call.args[0].checked_type.shape
    if len(in_shape0) != 4:
        return False

    yield  # The Compass OP Spec check code must be placed after this statement.

    res = []
    res.append(_check_shape(in_shape0, 0, 0, 4))
    res.append(1 <= call.attrs.block_size <= 16)

    return all(res)


@ir.register_op_attr("nn.depth_to_space", "target.aipu_compass.qnn")
@ir.register_op_attr("nn.depth_to_space", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape0 = call.args[0].checked_type.shape
    if len(in_shape0) != 4:
        return False

    yield  # The Compass OP Spec check code must be placed after this statement.

    res = []
    res.append(_check_shape(in_shape0, 0, 1, 4))
    res.append(1 <= call.attrs.block_size <= 16)

    return all(res)


@ir.register_op_attr("gather", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape0 = call.args[0].checked_type.shape
    in_shape1 = call.args[1].checked_type.shape

    res = []
    res.append(_check_dim([in_shape0, in_shape1], [1, 2, 3, 4, 5]))

    return all(res)


@ir.register_op_attr("gather_nd", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape0 = _convert_array(call.args[0].checked_type.shape)
    in_shape1 = _convert_array(call.args[1].checked_type.shape)

    rank_input0 = len(in_shape0)
    rank_input1 = len(in_shape1)
    batch_dims = int(call.attrs.batch_dims)

    res = []
    res.append(_check_dim([in_shape0, in_shape1], [2, 3, 4, 5], False))

    res.append(
        batch_dims < rank_input0
        and batch_dims < rank_input1
        and in_shape1[-1] <= rank_input0 - batch_dims
        # and in_shape0[:batch_dims] == in_shape1[:batch_dims]
    )

    res.append(isinstance(call.args[1], relay.Constant))

    return all(res)


@ir.register_op_attr("vision.non_max_suppression", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    attrs = call.attrs
    data, _, _, max_output_size, iou_threshold = call.args
    in_shape0 = data.checked_type.shape

    res = []
    res.append(isinstance(max_output_size, relay.Constant))
    res.append(isinstance(iou_threshold, relay.Constant))
    res.append(in_shape0[-1] == 5)
    res.append(attrs.top_k == -1)
    res.append(attrs.return_indices == 0)
    res.append(attrs.invalid_to_bottom == 0)
    res.append(attrs.coord_start == 1)
    res.append(attrs.score_index == 0)
    res.append(attrs.id_index < 0)
    if not all(res):
        return False

    yield  # The Compass OP Spec check code must be placed after this statement.
    return True


@ir.register_op_attr("erf", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape = call.args[0].checked_type.shape

    res = []
    # support 2, 3, 4 dim
    res.append(_check_dim([in_shape], [2, 3, 4]))
    # check shape of input
    res.append(_check_shape(in_shape, 0, 0, 4))

    return all(res)


@ir.register_op_attr("trunc", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.
    in_shape = call.args[0].checked_type.shape

    res = []
    # support 2, 3, 4 dim
    res.append(_check_dim([in_shape], [2, 3, 4]))
    # check shape of input
    res.append(_check_shape(in_shape, 0, 0, 4))

    return all(res)


@ir.register_op_attr("cumprod", "target.aipu_compass")
@ir.register_op_attr("cumsum", "target.aipu_compass")
@_checker
def _check_cumulate(call: relay.Call):
    in_shape = call.args[0].checked_type.shape
    out_shape = call.checked_type.shape
    attrs = call.attrs

    res = []
    # support 2, 3, 4, 5 dim
    res.append(_check_dim([in_shape, out_shape], [2, 3, 4, 5], False))
    # check reverse and exslusive
    dim = len(in_shape)
    res.append(attrs.axis in list(range(-dim, dim)))
    res.append(attrs.exclusive in [0, 1])

    return all(res)


@ir.register_op_attr("vision.get_valid_counts", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.

    in_shape0 = call.args[0].checked_type.shape
    thres = call.args[1]
    if not isinstance(thres, relay.Constant):
        return False

    yield
    res = []

    # relay op only support 3 dim input
    # no need to check dim
    res.append(_check_range(in_shape0[:1], [32]))
    res.append(_check_range(in_shape0[1:2], [16384]))
    res.append(_check_range(in_shape0[2:], [6], min_value=5))
    return all(res)


@ir.register_op_attr("vision.multibox_transform_loc", "target.aipu_compass")
@_checker
def _check(call: relay.Call):
    # Check if it is supported by AIPU Compass.

    attrs = call.attrs
    in_shape0 = call.args[0].checked_type.shape
    if not attrs.clip:
        return False

    yield
    res = []
    res.append(_check_range(in_shape0[:1], [32]))
    res.append(_check_range(in_shape0[1:2], [100]))
    res.append(_check_range(in_shape0[2:], [5000]))

    return all(res)
