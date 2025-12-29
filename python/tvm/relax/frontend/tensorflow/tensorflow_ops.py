# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
# pylint: disable=invalid-name, unused-argument, broad-except
"""TF: Tensorflow frontend."""
import numpy as np
import tensorflow as tf
import tvm
from tvm import relax
from tvm.topi.utils import get_const_tuple
from ... import op as _op


def get_relax_op(op_name):
    """Get the callable function from Relax based on operator name.
    Parameters
    ----------
    op_name : str
        The Relax operator name.
    """
    if "." in op_name:
        # explicit hierarchical modules
        op = _op
        try:
            for opn in op_name.split("."):
                op = getattr(op, opn)
        except AttributeError:
            op = None
    else:
        # try search op in various modules
        for candidate in (_op, _op.nn, _op.image, _op.vision):
            op = getattr(candidate, op_name, None)
            if op is not None:
                break
    if not op:
        raise tvm.error.OpNotImplemented(f"Unable to map op_name {op_name} to relax")
    return op


class AttrCvt(object):
    """Common attribute converter. An AttrConverter instance is a callable:
    ```
    attr_converter = AttrConverter(op_name, transforms={'a':'b', 'c':('d', 1)})
    new_op_name, new_attr = attr_converter(attrs)
    ```

    Parameters
    ----------
    op_name : str or callable
        If set as str, returned operator name is the str.
        If set as callable, returned operator is the str returned by calling:
        `op_name = func(attr)`

    transforms : dict of `new_name, or (new_name, default_value, transform function)`
        If only a new_name is provided, it's like renaming the attribute name.
        If default_value if provided, then the attribute is considered as optional.
        If transform function is provided, the original attribute value is handled
        by transform function.

    excludes : list
        A list of excluded attributes that should `NOT` appear.
        Raise NotImplementedError if occurred.

    disables : list
        A list of attributes that is disabled in relax.

    ignores : list
        A list of attributes that is ignored in relax.

    extras : dict
        A series of additional attributes should be added anyway to the returned
        attribute dict.

    custom_check : callable
        A custom function takes attribute, and return True/False.
        Raise RuntimeError if not bool(True) returned.
    """

    def __init__(
        self,
        op_name,
        transforms=None,
        excludes=None,
        disables=None,
        ignores=None,
        extras=None,
        custom_check=None,
    ):
        self._op_name = op_name
        self._transforms = transforms if transforms else {}
        self._excludes = excludes if excludes else []
        self._disables = disables if disables else []
        self._ignores = ignores if ignores else []
        self._extras = extras if extras else {}
        self._custom_check = custom_check

    def __call__(self, inputs, attrs, *args):
        self._ignores.append("_output_shapes")
        self._ignores.append("_input_shapes")
        self._ignores.append("T")
        self._ignores.append("use_cudnn_on_gpu")
        self._ignores.append("_node_name")
        self._ignores.append("is_training")
        self._ignores.append("_target_layout")

        # apply custom check
        if self._custom_check:
            func, msg = self._custom_check
            if not func(attrs):
                raise RuntimeError(f"Check failed: {msg}")
        # get new op_name
        if isinstance(self._op_name, str):
            op_name = self._op_name
        else:
            assert callable(self._op_name), "op_name can either be string or callable"
            op_name = self._op_name(attrs)

        # ignore 'tvm_custom' always
        self._ignores.append("tvm_custom")

        # convert attributes
        new_attrs = {}
        for k in attrs.keys():
            if k in self._excludes:
                raise NotImplementedError(f"Attribute {k} in operator {op_name} is not supported.")
            if k in self._disables:
                pass
                # TODO(compass-team): del this print statement or replace with logging
                # print(f"Attribute {k} is disabled in relax {op_name}")
            elif k in self._ignores:
                if k != "tvm_custom":
                    pass
                    # TODO(compass-team): del this print statement or replace with logging
                    # print(f"Attribute {k} is ignored in relax {op_name}")
            elif k in self._transforms:
                new_name, defaults, transform = self._parse_default(self._transforms[k])
                if defaults is None:
                    new_attr = self._required_attr(attrs, k)
                else:
                    new_attr = attrs.get(k, None)
                if new_attr is None:
                    new_attrs[new_name] = defaults
                else:
                    new_attrs[new_name] = transform(new_attr)
            else:
                # copy
                new_attrs[k] = attrs[k]
        # add extras
        new_attrs.update(self._extras)
        return get_relax_op(op_name)(*inputs, **new_attrs)

    def _parse_default(self, target):
        """Helper function to parse default values."""
        if not isinstance(target, (list, tuple)):
            k, v, t = target, None, lambda x: x
        elif len(target) == 1:
            k, v, t = target[0], None, lambda x: x
        elif len(target) == 2:
            k, v, t = target[0], target[1], lambda x: x
        elif len(target) > 2:
            k, v, t = target[0], target[1], target[2]
        else:
            k = None  # should raise
        if not isinstance(k, str):
            msg = f"{target} is not a valid target, (name, default) expected."
            raise ValueError(msg)
        return k, v, t

    def _parse_bool(self, value):
        """Helper function to parse default boolean values."""
        if isinstance(value, str):
            return value.strip().lower() in ["true", "1", "t", "y", "yes"]
        return bool(value)

    def _required_attr(self, attr, key):
        """Wrapper for getting required attributes."""
        assert isinstance(attr, dict)
        if key not in attr:
            raise AttributeError(f"Required attribute {key} not found.")
        return attr[key]


class StaticTensorArrayOps(object):
    """Contains tensor array related ops for fixed rank tensor array"""

    def __init__(self, prelude, dtype, shape, batch_dim=None):
        """Create tensor array ops registry"""
        raise NotImplementedError("not supported yet.")


def check_symbolic_shape(shape):
    return not all([isinstance(dim, (int, tvm.tir.IntImm)) for dim in shape])


def list_shape_of(tensor, ndim):
    shape_tensor = relax.op.nn.shape_of(tensor)
    return [
        relax.op.nn.strided_slice(shape_tensor, begin=[i], end=[i + 1], strides=[1])
        for i in range(ndim)
    ]


def _get_pad_pair(input1d, kernel1d, stride1d):
    msg = "SAME padding is not supported in combination with"
    msg += " dynamic height or width when stride is not 1."
    if not isinstance(input1d, (int, tvm.tir.IntImm)) and stride1d != 1:
        raise tvm.error.OpAttributeUnImplemented(msg)
    if stride1d == 1 or input1d % stride1d == 0:
        pad = max(kernel1d - stride1d, 0)
    else:
        pad = max(kernel1d - (input1d % stride1d), 0)

    pad_before = pad // 2
    pad_after = pad - pad_before

    return [pad_before, pad_after]


def _get_conv_transpose_pad_pair(input1d, output1d, kernel1d, stride1d):
    pad = max((input1d - 1) * stride1d + kernel1d - output1d, 0)

    pad_before = pad // 2
    pad_after = pad - pad_before

    return [pad_before, pad_after]


def _dimension_picker(prefix, surfix=""):
    def _impl(attr):
        kernel = attr["kernel_shape"]
        if len(kernel) == 2:
            return prefix + "2d" + surfix
        if len(kernel) == 3:
            return prefix + "3d" + surfix
        raise tvm.error.OpAttributeInvalid(
            f"Only 2D or 3D kernels are supported for operator {prefix}2d or 3d"
        )

    return _impl


def _dimension_constraint():
    def _dim_check(attrs):
        if len(attrs["kernel_shape"]) in (2, 3):
            return True
        return False

    return _dim_check, "Only 2d or 3d kernel supported."


def _get_param(params, input_node):
    if isinstance(input_node, relax.Constant):
        return np.atleast_1d(input_node.data.numpy())
    return params[input_node.name_hint].numpy()


def _get_num_param(params, input_node):
    return _get_param(params, input_node).item()


def _get_list_param(params, input_node, bb):
    try:
        return _get_param(params, input_node).tolist()
    except (IndexError, KeyError, AttributeError):
        return _infer_value(input_node, params, bb).numpy().tolist()


def _get_tuple_param(params, input_node):
    return tuple(_get_param(params, input_node))


def _get_more_static_shape(shape0, shape1):
    """Compare two shapes with the same rank,
    and return the one with fewer symbolic dimension.
    """
    assert len(shape0) == len(shape1)
    num_sym_dim0 = 0
    num_sym_dim1 = 0
    for dim0, dim1 in zip(list(shape0), list(shape1)):
        if not isinstance(dim0, int):
            num_sym_dim0 += 1
        if not isinstance(dim1, int):
            num_sym_dim1 += 1

    if num_sym_dim0 < num_sym_dim1:
        return shape0
    return shape1


def _rsqrt():
    def _impl(inputs, attr, params, bb):
        inputs.append(tvm.relax.const(-0.5, attr["T"].name))
        return AttrCvt(op_name="power")(inputs, attr)

    return _impl


def _argx(func, func_name):
    """A common wrapper for argmin and argmax operations"""

    def _impl(inputs, attr, params, bb):
        try:
            # In Tensorflow, `axis` argument is a Tensor, not attribute. We
            # support the case where it inputs from a scalar constant.
            axis_input_value = _get_num_param(params, inputs[1])
        except (IndexError, KeyError):
            raise TypeError(f"Unsupported argument for `{func_name}` : `axis` should be a constant")
        out = func(inputs[0], axis=axis_input_value, keepdims=False)
        dtype = attr["output_type"].name
        if dtype != "int32":
            out = relax.op.astype(out, dtype)
        return out

    return _impl


def _elemwise(name):
    def _impl(inputs, attr, params, bb):
        assert len(inputs) == 2, f"{name} take 2 inputs, {len(inputs)} given"
        if name == "multiply" and all(isinstance(inp, relax.Constant) for inp in inputs):
            output = np.multiply(inputs[0].data.numpy(), inputs[1].data.numpy())
            return relax.const(output, inputs[0].struct_info.dtype)

        return get_relax_op(name)(*inputs)

    return _impl


def _pooling(name):
    def _impl(inputs, attr, params, bb):

        attr["data_format"] = attr["data_format"].decode("utf-8")
        flip_layout = False

        input_shape = list(inputs[0].struct_info.shape)

        if attr["data_format"] == "NHWC":
            attr["kernel_shape"] = (attr["ksize"][1], attr["ksize"][2])
            attr["strides"] = (attr["strides"][1], attr["strides"][2])
        elif attr["data_format"] == "NCHW":
            attr["kernel_shape"] = (attr["ksize"][2], attr["ksize"][3])
            attr["strides"] = (attr["strides"][2], attr["strides"][3])
        else:
            msg = (
                f'Value {attr["data_format"]} of attribute "data_format" of operator Pooling '
                f"is not valid."
            )
            raise tvm.error.OpAttributeInvalid(msg)

        # if attr["_target_layout"] == "NCHW" and attr["data_format"] == "NHWC":
        #     tmp_shape = _infer_shape(inputs[0], bb)
        #     input_shape = [tmp_shape[ii] for ii in (0, 3, 1, 2)]
        #     inputs[0] = relax.op.nn.transpose(inputs[0], axes=(0, 3, 1, 2))
        #     attr["data_format"] = "NCHW"
        #     flip_layout = True

        # Fix padding
        attr["padding"] = attr["padding"].decode("utf-8")

        if attr["padding"] == "VALID":
            attr["padding"] = [0, 0]
        elif attr["padding"] == "SAME":
            stride_h, stride_w = attr["strides"]
            kernel_h, kernel_w = attr["kernel_shape"]
            if attr["data_format"] == "NHWC":
                in_h = input_shape[1]
                in_w = input_shape[2]
            else:
                in_h = input_shape[2]
                in_w = input_shape[3]

            pad_v = _get_pad_pair(in_h, kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, kernel_w, stride_w)

            attr["padding"] = [pad_v[0], pad_h[0], pad_v[1], pad_h[1]]
        elif attr["padding"] == "EXPLICIT":
            paddings = attr["explicit_paddings"]
            assert len(paddings) == 8
            if flip_layout or attr["data_format"] == "NHWC":
                attr["padding"] = [paddings[2], paddings[4], paddings[3], paddings[5]]
            else:
                attr["padding"] = [paddings[4], paddings[6], paddings[5], paddings[7]]
        else:
            msg = (
                f'Value {attr["padding"]} in attribute "padding" of operator Pooling is '
                f"not valid."
            )
            raise tvm.error.OpAttributeInvalid(msg)

        if name == "avg_pool":
            attr["count_include_pad"] = False

        out = AttrCvt(
            op_name=_dimension_picker(name),
            transforms={"kernel_shape": "pool_size", "data_format": "layout"},
            ignores=["ksize", "explicit_paddings"],
            extras={"ceil_mode": False},
            custom_check=_dimension_constraint(),
        )(inputs, attr)

        if flip_layout:
            out = relax.op.nn.transpose(out, axes=(0, 2, 3, 1))

        return out

    return _impl


def _conv(opname):
    def _impl(inputs, attr, params, bb):
        assert opname in ("conv", "conv_transpose", "depthwise")
        attr["data_format"] = attr["data_format"].decode("utf-8")
        flip_layout = False

        if opname == "conv_transpose" and attr["data_format"] == "NHWC":
            # transform to NCHW for TVM backend compatible and set 'flip_layout'
            # to have output flip back to NHWC
            inputs[2] = relax.op.permute_dims(inputs[2], axes=(0, 3, 1, 2))
            inputs[2] = bb.normalize(inputs[2])
            attr["strides"][1], attr["strides"][2], attr["strides"][3] = (
                attr["strides"][3],
                attr["strides"][1],
                attr["strides"][2],
            )
            attr["data_format"] = "NCHW"

            # Check whether output shapes attribute is set and not None
            if (
                opname == "conv_transpose"
                and len(attr["_output_shapes"]) > 0
                and attr["_output_shapes"][0]
            ):
                tmp_shape = attr["_output_shapes"][0]
                tmp_shape = [tmp_shape[ii] for ii in (0, 3, 1, 2)]
                attr["_output_shapes"][0] = tmp_shape

            flip_layout = True

        inputs_data = inputs[0] if opname != "conv_transpose" else inputs[2]
        # NCHW Layout require weights transpose
        weights_shape = list(inputs[1].struct_info.shape)
        if attr["data_format"] == "NCHW":
            tmp_shape = weights_shape
            if opname in ["conv", "conv_transpose"]:
                tmp_shape = [tmp_shape[ii] for ii in (3, 2, 0, 1)]
                inputs[1] = relax.op.permute_dims(inputs[1], axes=(3, 2, 0, 1))
            else:
                tmp_shape = [tmp_shape[ii] for ii in (2, 3, 0, 1)]
                inputs[1] = relax.op.permute_dims(inputs[1], axes=(2, 3, 0, 1))
            weights_shape = tmp_shape

        input_shape = list(inputs_data.struct_info.shape)
        if attr["data_format"] == "NHWC":
            in_channels = input_shape[3]
            kernel_h, kernel_w, kernel_factor, depth_mult = weights_shape
            attr["kernel_shape"] = (weights_shape[0], weights_shape[1])
            if opname == "conv":
                attr["channels"] = weights_shape[3]
            elif opname == "conv_transpose":
                attr["channels"] = weights_shape[2]
            else:
                attr["channels"] = input_shape[3] * depth_mult
                if int(kernel_factor) != 1:
                    new_shape = (int(kernel_h), int(kernel_w), 1, int(attr["channels"]))  # HWIO
                    inputs[1] = relax.const(np.reshape(inputs[1].data.numpy(), new_shape))

            if "dilations" in attr:
                attr["dilations"] = (attr["dilations"][1], attr["dilations"][2])
            attr["strides"] = (attr["strides"][1], attr["strides"][2])
        elif attr["data_format"] == "NCHW":
            in_channels = input_shape[1]
            _, depth_mult, kernel_h, kernel_w = weights_shape
            attr["kernel_shape"] = (weights_shape[2], weights_shape[3])
            if opname == "conv":
                attr["channels"] = weights_shape[0]
            elif opname == "conv_transpose":
                attr["channels"] = weights_shape[1]
            else:
                attr["channels"] = input_shape[1] * depth_mult
                if attr["channels"] < 0:
                    attr["channels"] *= -1

            if "dilations" in attr:
                attr["dilations"] = (attr["dilations"][2], attr["dilations"][3])
            attr["strides"] = (attr["strides"][2], attr["strides"][3])
        else:
            msg = (
                f'Value {attr["data_format"]} in attribute "data_format" of operator Conv is '
                f"not valid."
            )
            raise tvm.error.OpAttributeInvalid(msg)

        if opname == "depthwise":
            attr["groups"] = int(in_channels)

        # Fix padding
        attr["padding"] = attr["padding"].decode("utf-8")

        if attr["padding"] == "VALID":
            attr["padding"] = [0, 0]
        elif attr["padding"] == "SAME":
            stride_h, stride_w = attr["strides"]
            kernel_h, kernel_w = attr["kernel_shape"]

            pdata_shape = input_shape
            # Check whether output shapes attribute is set and not None
            if (
                opname == "conv_transpose"
                and len(attr["_output_shapes"]) > 0
                and attr["_output_shapes"][0]
            ):
                pdata_shape = attr["_output_shapes"][0]

            if attr["data_format"] == "NHWC":
                in_h = pdata_shape[1]
                in_w = pdata_shape[2]
            else:
                in_h = pdata_shape[2]
                in_w = pdata_shape[3]

            dilation_h = attr["dilations"][0]
            dilation_w = attr["dilations"][1]
            dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
            dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
            # when op is conv_transpose, the pad is calculated by output shape.
            if opname == "conv_transpose":
                out_shape = inputs[0].data.numpy()
                out_h = out_shape[1]
                out_w = out_shape[2]
                pad_v = _get_conv_transpose_pad_pair(in_h, out_h, dilated_kernel_h, stride_h)
                pad_h = _get_conv_transpose_pad_pair(in_w, out_w, dilated_kernel_w, stride_w)
            else:
                pad_v = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
                pad_h = _get_pad_pair(in_w, dilated_kernel_w, stride_w)

            attr["padding"] = [pad_v[0], pad_h[0], pad_v[1], pad_h[1]]
        elif attr["padding"] == "EXPLICIT":
            paddings = attr["explicit_paddings"]
            assert len(paddings) == 8
            if flip_layout or attr["data_format"] == "NHWC":
                attr["padding"] = [paddings[2], paddings[4], paddings[3], paddings[5]]
            else:
                attr["padding"] = [paddings[4], paddings[6], paddings[5], paddings[7]]
        else:
            msg = (
                f'Value {attr["padding"]} in attribute "padding" of operator Conv is not ' f"valid."
            )
            raise tvm.error.OpAttributeInvalid(msg)

        if "kernel_layout" not in attr:
            if opname in ("conv", "depthwise"):
                attr["kernel_layout"] = "HWIO" if attr["data_format"] == "NHWC" else "OIHW"
            else:
                # conv_transpose has weights be IOHW, because the attr["data_format"] always be NCHW
                attr["kernel_layout"] = "IOHW"

        # Ignore the new attributes from TF2.0, for now.
        out = AttrCvt(
            op_name=_dimension_picker(
                "conv", surfix="_transpose" if opname == "conv_transpose" else ""
            ),
            ignores=["explicit_paddings", "kernel_shape", "channels"],
            transforms={
                "data_format": "data_layout",
                "dilations": ("dilation", (0, 0)),
                "group": ("groups", 1),
            },
            custom_check=_dimension_constraint(),
        )([inputs_data, inputs[1]], attr)

        if flip_layout:
            out = relax.op.permute_dims(out, axes=(0, 2, 3, 1))

        return out

    return _impl


def _nms(return_scores=False):
    def _impl(inputs, attr, params, bb):
        # TODO(compass-team): Support later.
        placeholder = np.ones(inputs[0].struct_info.shape, "int32")
        return relax.const(placeholder)

        # # Get parameter values
        # try:
        #     max_output_size = int(np.atleast_1d(inputs[2].data.numpy().astype("int64"))[0])
        # except Exception:
        #     try:
        #         max_output_size = (
        #             _infer_value(inputs[2], params, bb).numpy().astype("int64").tolist()[0]
        #         )
        #     except Exception:
        #         max_output_size = inputs[2]
        # iou_threshold = np.atleast_1d(inputs[3].data.numpy())[0]
        # # score_threshold was introduced from V3
        # score_threshold = np.atleast_1d(inputs[4].data.numpy())[0] if len(inputs) > 4 else 0.0
        # pad_output = "pad_to_max_output_size"

        # # Generate data with shape (1, num_anchors, 5)
        # scores = AttrCvt(
        #     op_name="expand_dims",
        #     ignores=["T_threshold", pad_output],
        #     extras={"axis": -1, "num_newaxis": 1},
        # )([inputs[1]], attr)
        # data = get_relay_op("concatenate")([scores, inputs[0]], -1)
        # data = get_relay_op("expand_dims")(data, 0, 1)

        # # reason why using get_valid_counts is for inference performance
        # ct, data, indices = get_relay_op("get_valid_counts")(
        #     data, score_threshold=score_threshold, id_index=-1, score_index=0
        # )
        # # TensorFlow NMS doesn't have parameter top_k
        # top_k = -1
        # # TF doesn't have class id for nms input
        # score_index = 0
        # nms_ret = get_relay_op("non_max_suppression")(
        #     data=data,
        #     valid_count=ct,
        #     indices=indices,
        #     max_output_size=max_output_size,
        #     iou_threshold=iou_threshold,
        #     force_suppress=True,
        #     top_k=top_k,
        #     coord_start=1,
        #     score_index=score_index,
        #     id_index=-1,
        #     return_indices=True,
        #     invalid_to_bottom=False,
        # )

        # if pad_output in attr and attr[pad_output]:
        #     return nms_ret
        # # squeeze it, TF NMS is not batched
        # size = get_relay_op("squeeze")(nms_ret[1], axis=[1])
        # data_slice = get_relay_op("squeeze")(nms_ret[0], axis=[0])

        # # slice to get the dynamic result
        # ret = get_relay_op("strided_slice")(
        #     data_slice, begin=relax.const([0]), end=size, slice_mode="size"
        # )

        # # NonMaxSuppressionV5 returns scores. pad_output is always False for NMSv5.
        # if return_scores:
        #     if "soft_nms_sigma" in attr and attr["soft_nms_sigma"] != 0.0:
        #         raise tvm.error.OpAttributeUnImplemented(
        #             "soft_nms_sigma for NonMaxSuppressionV5 is not supported"
        #         )
        #     ret_scores = relax.op.nn.take(inputs[1], ret, axis=0)
        #     return relax.TupleWrapper(relax.Tuple([ret, ret_scores, size]), 3)

        # return ret

    return _impl


def _one_hot():
    def _impl(inputs, attr, params, bb):
        depth = int(_get_num_param(params, inputs[1]))
        on_value = _get_num_param(params, inputs[2])
        off_value = _get_num_param(params, inputs[3])
        new_inputs = [
            inputs[0],
            relax.PrimValue(on_value),
            relax.PrimValue(off_value),
        ]
        return AttrCvt("one_hot", ignores=["TI"], extras={"depth": depth})(new_inputs, attr)

    return _impl


def convert_combined_nms_with_all_class_nms(
    batch_size,
    max_output_boxes_per_batch,
    num_class,
    boxes,
    scores,
    max_output_boxes_per_class,
    iou_threshold,
    score_threshold,
    max_total_size,
    clip_boxes,
):
    """Converts TF combined_nms using Relay all_class_max_suppression op"""
    (
        selected_indices,
        selected_scores,
        num_detections,
    ) = relax.op.nn.vision.all_class_non_max_suppression(
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        output_format="tensorflow",
    )
    box_range = relax.op.nn.arange(
        relax.op.nn.const(0, dtype="int64"),
        relax.op.nn.const(max_total_size, dtype="int64"),
        dtype="int64",
    )
    assert isinstance(batch_size, int), "dynamic batch size not supported yet."
    tile_batch_reps = relax.op.nn.const([batch_size, 1])
    box_range_2d = relax.op.nn.tile(box_range, tile_batch_reps)
    valid_mask = relax.op.nn.cast(
        relax.op.nn.less(box_range_2d, relax.op.nn.expand_dims(num_detections, axis=1)), "float32"
    )

    def select_topk(do_zero_pad):
        def true_branch():
            arange = relax.op.nn.arange(
                relax.op.nn.const(0, dtype="int64"),
                relax.op.nn.const(max_output_boxes_per_batch, dtype="int64"),
                dtype="int64",
            )
            pad = relax.op.nn.full(
                relax.op.nn.const(0, dtype="int64"), (max_total_size - max_output_boxes_per_batch,)
            )
            topk_indices = relax.op.nn.tile(
                relax.op.nn.concatenate([arange, pad], 0), tile_batch_reps
            )
            nmsed_scores = relax.op.nn.gather(selected_scores, 1, topk_indices)
            nmsed_scores = nmsed_scores * valid_mask
            return nmsed_scores, topk_indices

        def false_branch():
            if isinstance(max_output_boxes_per_class, int):
                # Do topk on smaller input if possible
                slice_mx = relax.op.nn.const(
                    [max_output_boxes_per_class * num_class], dtype="int64"
                )
                selected_scores_slice = relax.op.nn.strided_slice(
                    selected_scores,
                    begin=relax.op.nn.const([0], dtype="int64"),
                    end=slice_mx,
                    axes=[1],
                )
            else:
                selected_scores_slice = selected_scores
            return relax.op.nn.topk(
                selected_scores_slice, k=max_total_size, axis=1, ret_type="both"
            )

        # TODO(masahi): support dynamic num_boxes
        # return relax.If(do_zero_pad, true_branch(), false_branch())
        return true_branch() if do_zero_pad else false_branch()

    assert isinstance(max_output_boxes_per_batch, int), "dynamic number of boxes not supported yet."
    nmsed_scores, topk_indices = select_topk(max_output_boxes_per_batch < max_total_size)

    indices = relax.op.nn.take(selected_indices, topk_indices, axis=1, batch_dims=1)
    nmsed_box_indices = relax.op.nn.take(indices, relax.op.nn.const(1), axis=2)
    nmsed_classes = relax.op.nn.take(indices, relax.op.nn.const(0), axis=2)
    nmsed_classes = relax.op.nn.cast(nmsed_classes, "float32")
    nmsed_boxes = relax.op.nn.take(boxes, nmsed_box_indices, axis=1, batch_dims=1)
    num_detections = relax.op.nn.minimum(
        num_detections, relax.op.nn.const(max_total_size, dtype="int64")
    )

    if clip_boxes:
        nmsed_boxes = relax.op.nn.maximum(nmsed_boxes, relax.const(0, dtype="float32"))
        nmsed_boxes = relax.op.nn.minimum(nmsed_boxes, relax.const(1, dtype="float32"))

    nmsed_boxes = nmsed_boxes * relax.op.nn.expand_dims(valid_mask, axis=2)

    return relax.TupleWrapper(
        relax.Tuple([nmsed_boxes, nmsed_scores, nmsed_classes, num_detections]), 4
    )


def _crop_and_resize():
    def _impl(inputs, attr, params, bb):
        # input image is a 4-D tensor of shape [batch, image_height, image_width, depth]
        # boxes is a 2-D tensor of shape [num_boxes, 4], 4 is for [y1, x1, y2, x2]
        crop_size = _get_list_param(params, inputs[3], bb)

        method = attr["method"].decode()
        method = "nearest_neighbor" if method == "nearest" else method
        if method not in ["bilinear", "nearest_neighbor"]:
            raise tvm.error.OpAttributeUnImplemented(f"Method {method} is not supported")
        layout = attr["layout"] if "layout" in attr else "NHWC"
        extrapolation_value = attr["extrapolation_value"]

        return get_relay_op("crop_and_resize")(
            inputs[0], inputs[1], inputs[2], crop_size, layout, method, extrapolation_value
        )

    return _impl


def _cast():
    def _impl(inputs, attr, params, bb):
        return inputs[0].astype(attr["DstT"].name)

    return _impl


def _expand_dims():
    def _impl(inputs, attr, params, bb):
        dim_input = inputs.pop(1)
        axis = _get_num_param(params, dim_input)
        if isinstance(inputs[0], relax.Constant):
            data_np = inputs[0].data.numpy()
            return relax.const(np.expand_dims(data_np, int(axis)), str(data_np.dtype))
        return AttrCvt(
            op_name="expand_dims",
            ignores=["Tdim", "N"],
            extras={"axis": int(axis)},
        )(inputs, attr)

    return _impl


def _resize(method):
    def _impl(inputs, attr, params, bb):
        if attr["_output_shapes"][0] is not None:
            size = attr["_output_shapes"][0][1:3]
            # Important that the size is defined. If an axis is not, we need to infer what
            # the shape should be.
            if -1 in size:
                size = inputs[1]
        else:
            size = inputs[1]

        size = inputs[1].data.numpy().astype("int64").tolist()
        attr["size"] = size
        inputs.pop(1)
        # NHWC
        attr["layout"] = "NHWC"
        if attr.pop("align_corners") is True:
            attr["coordinate_transformation_mode"] = "align_corners"
        else:
            attr["coordinate_transformation_mode"] = "asymmetric"

        # Ignore the new attributes from TF2.0, for now.
        return AttrCvt(
            op_name="resize2d",
            ignores=["Tdim", "half_pixel_centers"],
            extras={"method": method, "roi": None},
        )(inputs, attr)

    return _impl


def _assert():
    # ToDo: In general people want asserts to be gone from TensorFlow graphs
    # when they are optimizing them, so converting it to a no-op is
    # reasonable. However, it would be nice to have the option to keep them
    # once Relay gets a Halt or Assert op.
    return _no_op()


def _no_op():
    def _impl(inputs, attr, params, bb):
        # ToDo: This should really be an op that returns nothing, which could
        # be represented as an empty tuple. It turns out that TVM
        # infrastructure doesn't like running functions that return None and
        # also don't like running functions that return an empty tuple. So it
        # doesn't work, but it should be made to work and then this could be
        # improved. In the mean time, it is hard to imagine a case where it
        # matters in any real way that a no-op is converted to a constant 0.
        return tvm.relay.const(0)

    return _impl


def _matmul():
    def _impl(inputs, attr, params, bb):
        if attr["transpose_a"]:
            inputs[0] = relax.op.permute_dims(inputs[0], axes=(1, 0))
        if attr["transpose_b"]:
            inputs[1] = relax.op.permute_dims(inputs[1], axes=(1, 0))
        return AttrCvt(
            op_name="matmul",
            ignores=["transpose_a", "transpose_b", "T"],
        )(inputs, attr)

    return _impl


def _batch_matmul():
    def _impl(inputs, attr, params, bb):
        input_x = inputs[0]
        input_y = inputs[1]
        orig_shape_x = input_x.struct_info.shape.values
        orig_shape_y = input_y.struct_info.shape.values
        ndim = len(orig_shape_x)
        ndim_y = len(orig_shape_y)

        is_static = not check_symbolic_shape(orig_shape_x)

        # reshape n-dimensional batch matmul into 3d
        if ndim > 3:
            outer_dims = [orig_shape_x[i] for i in range(0, len(orig_shape_x) - 2)]
            if is_static:
                num_outer_elts = np.prod(outer_dims)
                new_shape_x = (num_outer_elts, orig_shape_x[-2], orig_shape_x[-1])
                if ndim_y > 2:
                    new_shape_y = (num_outer_elts, orig_shape_y[-2], orig_shape_y[-1])
                elif ndim_y == 2:
                    new_shape_y = (1, orig_shape_y[-2], orig_shape_y[-1])
            else:  # handle dynamic shape (dyn.reshape op)
                shape_of_x = list_shape_of(inputs[0], ndim)
                shape_of_y = list_shape_of(inputs[1], ndim)
                new_shape_x = [relax.op.nn.const(1), shape_of_x[-2], shape_of_x[-1]]
                new_shape_y = [relax.op.nn.const(1), shape_of_y[-2], shape_of_y[-1]]
                for i in range(ndim - 2):
                    new_shape_x[0] *= shape_of_x[i]
                    new_shape_y[0] *= shape_of_y[i]
                new_shape_x = relax.op.nn.concatenate(relax.op.nn.Tuple(new_shape_x), axis=0)
                new_shape_y = relax.op.nn.concatenate(relax.op.nn.Tuple(new_shape_y), axis=0)

            input_x = relax.op.reshape(input_x, new_shape_x)
            input_y = relax.op.reshape(input_y, new_shape_y)
        elif ndim_y == 2:
            input_y = relax.op.reshape(input_y, (1, orig_shape_y[-2], orig_shape_y[-1]))
        adj_x = attr["adj_x"]
        adj_y = attr["adj_y"]

        # Strictly convert all batch_matmul to NT format
        input_x = relax.op.permute_dims(input_x, axes=[0, 2, 1]) if adj_x else input_x
        input_y = relax.op.permute_dims(input_y, axes=[0, 2, 1]) if adj_y else input_y
        ret = relax.op.matmul(input_x, input_y)

        # reshape result back to n-dimensional
        if ndim > 3:
            if is_static:
                final_shape = list(orig_shape_x)
                final_shape[-2] = orig_shape_x[-1] if adj_x else orig_shape_x[-2]
                final_shape[-1] = orig_shape_y[-2] if adj_y else orig_shape_y[-1]
            else:
                # calculate the resulting shape = [shape[:-2], 0, 0]
                final_shape = list(shape_of_x)
                final_shape[-2] = shape_of_x[-1] if adj_x else shape_of_x[-2]
                final_shape[-1] = shape_of_y[-2] if adj_y else shape_of_y[-1]
                final_shape = relax.op.nn.concatenate(relax.op.nn.Tuple(final_shape), axis=0)

            ret = relax.op.reshape(ret, final_shape)
        return ret

    return _impl


def row_wise_divide(multi_dim_tensor, one_dim_vector):
    """
    This function enables row-wise division of multi_dim_tensor and one_dim_vector.
    To achieve this, it is first tiled to the appropriate shape and then elemwise_division
    """
    multi_dim_tensor_offrow_shape = relax.op.nn.strided_slice(
        relax.op.nn.shape_of(multi_dim_tensor, "int32"), [1], [-1], slice_mode="size"
    )
    one_dim_vector_tiled_shape = relax.op.nn.concatenate(
        [relax.op.nn.reverse(multi_dim_tensor_offrow_shape, 0), relax.const([1])], axis=0
    )
    one_dim_vector_tiled = relax.op.nn.transpose(
        relax.op.nn.tile(one_dim_vector, one_dim_vector_tiled_shape)
    )
    return relax.op.nn.divide(multi_dim_tensor, one_dim_vector_tiled)


def count_all_indices(segment_ids, counts_dtype, num_segments=None):
    """
    This snippet calculates the sqrt count of each index among all valid indices
    Valid indices are from 0 to max of [segment ids, num_segments]
    """

    max_segments = relax.op.nn.reshape(relax.op.nn.max(segment_ids), -1) + relax.const([1])
    if num_segments:
        max_segments = relax.op.nn.maximum(max_segments, relax.const([num_segments]))
    max_ones = relax.op.nn.maximum(max_segments, relax.op.nn.shape_of(segment_ids))
    counts = relax.op.nn.segment_sum(
        relax.op.nn.ones(max_ones, counts_dtype), segment_ids, num_segments=num_segments
    )
    real_counts = relax.op.nn.clip(counts, 1, 2147483647)  # Clip max doesn't work over int32
    return real_counts


def _identity():
    def _impl(inputs, attr, params, bb):
        return inputs[0]

    return _impl


def _identityn():
    def _impl(inputs, attr, params, bb):
        return inputs[0]

    return _impl


def _concatV2():
    def _impl(inputs, attr, params, bb):
        pop_node = inputs.pop(len(inputs) - 1)
        try:
            axis = int(_get_num_param(params, pop_node))
        except (IndexError, KeyError, AttributeError):
            try:
                axis = int(_infer_value(pop_node, params, bb).numpy())
            except Exception:
                axis = int(pop_node)

        # If all inputs are constant, perform computation directly.
        if all(isinstance(inp, relax.Constant) for inp in inputs):
            const_inputs = []
            for inp in inputs:
                const_inputs.append(inp.data.numpy())
            out = np.concatenate(const_inputs, axis=axis)
            dtype = inputs[0].struct_info.dtype
            return relax.const(out, dtype)

        is_all_inputs_dim_eq_0 = all(x.struct_info.ndim == 0 for x in inputs)
        if is_all_inputs_dim_eq_0:
            return relax.op.stack(inputs)

        return AttrCvt(op_name="concat", ignores=["T", "N", "Tidx"], extras={"axis": axis})(
            [inputs], attr
        )

    return _impl


def _pack():
    def _impl(inputs, attr, params, bb):
        axis = int(attr["axis"])
        if all(isinstance(i, relax.Constant) for i in inputs):
            inputs_reshaped = [np.expand_dims(i.data.numpy(), axis=axis) for i in inputs]
            out = np.concatenate(inputs_reshaped, axis)
            return out
        inputs_reshaped = [relax.op.expand_dims(i, axis=axis) for i in inputs]
        return relax.op.concat(inputs_reshaped, axis)

    return _impl


def _tensor_array():
    def _impl(inputs, attr, params, bb):
        dtype_str = attr.get("dtype").name
        assert not attr["dynamic_size"], "Dynamic size tensor array is " "not supported in TVM yet."
        assert "shape" in attr
        assert isinstance(inputs[0], relax.Constant), "Need to support dynamic size."
        shape = [int(inputs[0].data.numpy())] + list(attr["shape"])
        return relax.op.full(shape, relax.const(0, dtype_str), dtype_str)

    return _impl


def _tensor_array_scatter():
    def _impl(inputs, attr, params, bb):
        if isinstance(inputs[1], relax.Constant):
            data_np = inputs[1].data.numpy()
            indices = relax.const(data_np.reshape([-1, 1]), str(data_np.dtype))
        else:
            indices = relax.op.reshape(inputs[1], [-1, 1])
        return relax.op.scatter_nd(inputs[0], indices, inputs[2])

    return _impl


def _tensor_array_gather():
    def _impl(inputs, attr, params, bb):
        return relax.op.take(inputs[2], inputs[1], axis=0)

    return _impl


def _tensor_array_size():
    def _impl(inputs, attr, params, bb):
        return relax.const(int(inputs[0].struct_info.shape[0]), "int32")

    return _impl


def _tensor_array_write():
    def _impl(inputs, attr, params, bb):
        if isinstance(inputs[1], relax.Constant):
            data_np = inputs[1].data.numpy()
            indices = relax.const(data_np.reshape([-1]), str(data_np.dtype))
        else:
            indices = relax.op.reshape(inputs[1], [-1])
        return relax.op.scatter_nd(inputs[0], indices, inputs[2])

    return _impl


def _tensor_array_read():
    def _impl(inputs, attr, params, bb):
        return relax.op.take(inputs[2], inputs[1], axis=0)

    return _impl


def _tile():
    def _impl(inputs, attr, params, bb):
        reps_input = inputs.pop()
        if isinstance(reps_input, relax.Call):
            np_reps = _infer_value(reps_input, params, bb).numpy()
            reps = [np_reps.flatten()[i] for i in range(np_reps.flatten().shape[0])]
        else:
            reps = _get_list_param(params, reps_input, bb)
        new_input = [inputs.pop(0)]

        return AttrCvt(op_name="tile", extras={"repeats": tuple(reps)}, ignores=["Tmultiples"])(
            new_input, attr
        )

    return _impl


def _slice():
    def _impl(inputs, attr, params, bb):
        try:
            begin = _get_list_param(params, inputs[1], bb)
        except Exception:
            # Handle symbolic begin
            begin = inputs[1]
        try:
            size = _get_list_param(params, inputs[2], bb)
        except Exception:
            # Handle symbolic size
            size = inputs[2]

        # Align begin and strides for dynamic shape.
        data_dim = len(_infer_shape(inputs[0], bb))
        strides = [1] * data_dim
        if not isinstance(begin, (relax.Call, relax.Var)):
            for _ in range(len(begin), data_dim):
                begin.append(0)
        elif not isinstance(size, (relax.Call, relax.Var)):
            for _ in range(len(size), data_dim):
                size.append(-1)
        return relax.op.nn.strided_slice(
            inputs[0], begin=begin, end=size, strides=strides, slice_mode="size"
        )

    return _impl


def _reshape():
    def _impl(inputs, attr, params, bb):
        pop_node = inputs.pop(1)
        shape_arg = _get_tuple_param(params, pop_node)

        return AttrCvt(op_name="reshape", extras={"shape": shape_arg}, ignores=["Tshape"])(
            inputs, attr
        )

    return _impl


def _sparse_to_dense():
    def _impl(inputs, attr, params, bb):
        # TODO(compass-team): Support later.
        return inputs[0]
        # sparse_indices = inputs[0]
        # output_shape = inputs[1]
        # sparse_values = inputs[2]
        # default_value = inputs[3]

        # return relax.op.nn.sparse_to_dense(
        #     sparse_indices, output_shape, sparse_values, default_value
        # )

    return _impl


def _bias_add():
    def _impl(inputs, attr, params, bb):
        # Must expand for proper broadcasting in NCHW.
        if "data_format" in attr and attr["data_format"].decode("utf-8") == "NCHW":
            bias = relax.op.nn.reshape(inputs[1], newshape=(1, -1, 1, 1))
        else:
            bias = inputs[1]
        return relax.op.add(inputs[0], bias)

    return _impl


def _squeeze():
    def _impl(inputs, attr, params, bb):
        if len(attr["squeeze_dims"]) == 0:
            attr["squeeze_dims"] = None
        return AttrCvt(
            op_name="squeeze", transforms={"squeeze_dims": "axis"}, ignores=["T", "_cloned"]
        )(inputs, attr)

    return _impl


def _fused_batch_norm():
    def _impl(inputs, attr, params, bb):
        # Tensorflow: (data, gamma, beta, moving_mean, moving_variance)
        # Relay:       (data, gamma, beta, moving_mean, moving_varience)
        assert len(inputs) == 5
        axis = 3
        need_cast = False

        if "data_format" in attr:
            attr["data_format"] = attr["data_format"].decode("utf-8")
            if attr["data_format"] == "NCHW":
                axis = 1
        if "U" in attr and attr["U"].name != attr["T"].name:
            need_cast = True
            inputs[0] = relax.op.nn.cast(inputs[0], dtype=attr["U"].name)
        # Check if mean and variance are empty
        # If so, replace them with Mean and Variance Ops
        # For run-time calculation
        moving_mean_shape = [int(n) for n in inputs[3].data.shape]
        moving_variance_shape = [int(n) for n in inputs[4].data.shape]
        if moving_mean_shape[0] == 0 and moving_variance_shape[0] == 0:
            inputs[3] = relax.op.nn.mean(inputs[0], axis=axis, keepdims=False, exclude=True)
            inputs[4] = relax.op.nn.variance(inputs[0], axis=axis, keepdims=False, exclude=True)
        out = AttrCvt(
            op_name="batch_norm",
            transforms={"scale_after_normalization": "scale", "variance_epsilon": "epsilon"},
            extras={"axis": axis},
            ignores=["data_format", "U", "exponential_avg_factor"],
            disables=["momentum"],
        )(inputs, attr)

        if need_cast:
            out = relax.TupleGetItem(out.astuple(), 0)
            out = relax.op.nn.cast(out, dtype=attr["T"].name)
        return out

    return _impl


def _relu6():
    def _impl(inputs, attr, params, bb):
        return relax.op.clip(inputs[0], min=0, max=6)

    return _impl


def _shape():
    def _impl(inputs, attr, params, bb):
        is_symbolic_shape = False
        input_sinfo = inputs[0].struct_info
        if hasattr(input_sinfo, "shape") and input_sinfo.shape is not None:
            input_shape = list(input_sinfo.shape)
            for axis in input_shape:
                if not isinstance(axis, (int, tvm.tir.IntImm)):
                    is_symbolic_shape = True
                    break
        else:
            is_symbolic_shape = True

        if is_symbolic_shape:
            ret = relax.op.shape_of(inputs[0])
        else:
            ret = np.array(input_shape, dtype=attr["out_type"].name)
        return ret

    return _impl


def _fill():
    def _impl(inputs, attr, params, bb):
        dtype = attr["T"].name
        output_shape = None
        is_shape_const = False
        if isinstance(inputs[0], relax.Call) and inputs[0].op.name.endswith("shape_of"):
            sinfo_ndim = inputs[0].struct_info.ndim
            var_name = f"fill_oshape_var_{inputs[0].args[0].name_hint}"
            output_shape = relax.Var(var_name, relax.ShapeStructInfo(ndim=sinfo_ndim))
        elif isinstance(inputs[0], relax.Constant):
            output_shape = inputs[0].data.numpy().tolist()
            is_shape_const = all(isinstance(x, int) for x in output_shape)

            if is_shape_const and isinstance(inputs[1], relax.Constant):
                fill_value = inputs[1].data.numpy().tolist()
                return relax.const(np.full(output_shape, fill_value, dtype), dtype)
        else:
            raise ValueError("Unsupported input type for 'inputs[0]'")

        return relax.op.full(output_shape, inputs[1], attr["T"].name)

    return _impl


def _sum():
    def _impl(inputs, attr, params, bb):
        axis = _get_tuple_param(params, inputs[1])
        return AttrCvt(
            op_name="sum",
            extras={"axis": axis},
            transforms={"keep_dims": "keepdims"},
            ignores=["name", "Tidx"],
        )([inputs[0]], attr)

    return _impl


def _reduce(op):
    def _impl(inputs, attr, params, bb):
        axis = _get_list_param(params, inputs[1], bb)
        axis = tuple(axis)
        if not axis:
            axis = None
        return AttrCvt(
            op_name=op,
            extras={"axis": axis},
            transforms={"keep_dims": "keepdims"},
            ignores=["name", "Tidx"],
        )([inputs[0]], attr)

    return _impl


def _square():
    def _impl(inputs, attr, params, bb):
        return relax.op.multiply(inputs[0], inputs[0])

    return _impl


def _gather():
    "GatherV2, Gather"

    def _impl(inputs, attr, params, bb):
        if len(inputs) > 2:
            axis = _get_num_param(params, inputs.pop(2))
        else:
            axis = 0
        new_input = inputs[0:2]
        if all(isinstance(x, relax.Constant) for x in new_input):
            np_inps = [x.data.numpy() for x in new_input]
            output = np.take(*np_inps, axis=axis)
            return relax.const(output, str(output.dtype))
        op_ = AttrCvt(
            op_name="take",
            extras={"axis": int(axis)},
            ignores=["Tindices", "Tparams", "validate_indices", "Taxis", "_class", "batch_dims"],
        )(new_input, attr)
        return op_

    return _impl


def _stridedSlice():
    def _impl(inputs, attr, params, bb):
        """Strided Slice.
        Operator description: https://www.tensorflow.org/api_docs/python/tf/strided_slice
        Tensorflow mask validation: https://github.com/tensorflow/tensorflow/blob/master/
        tensorflow/core/util/strided_slice_op.cc#L147-L368
        """
        begin = _get_list_param(params, inputs[1], bb)
        end = _get_list_param(params, inputs[2], bb)
        stride = _get_list_param(params, inputs[3], bb)

        begin_mask = int(attr.get("begin_mask", 0))
        end_mask = int(attr.get("end_mask", 0))
        ellipsis_mask = int(attr.get("ellipsis_mask", 0))
        new_axis_mask = int(attr.get("new_axis_mask", 0))
        shrink_axis_mask = int(attr.get("shrink_axis_mask", 0))
        in_type = inputs[0].struct_info
        data_shape = get_const_tuple(in_type.shape.values)
        data_dim = len(data_shape)
        stride_dim = len(stride)
        if data_dim == 0 and isinstance(inputs[0], relax.Constant):
            new_data = inputs[0].data.numpy().reshape(1)
            return relax.const(new_data, inputs[0].data.dtype)

        # This is a special routine to handle strided_slice after shape_of.
        # We need this since in some cases we want to do strided_slice on
        # a partial symbolic shape, such as (1, ?), and get a static shape
        # (1,). Directly slice on shape_of will result in fully dynamic shape.
        # TODO(kevinthesun): Can we generalize this process with partial eval?
        if isinstance(inputs[0], relax.Call) and inputs[0].op.name == "relax.shape_of":
            raise NotImplementedError("Not support shape_of + stride_slice yet.")
            # bg = begin[0]
            # ed = end[0]
            # st = stride[0]

            # if ed <= 0 < st:
            #     ed += data_shape[0]

            # in_shape = _infer_shape(inputs[0].args[0], bb)
            # dtype = in_type.checked_type.dtype
            # out_data = []
            # idx = bg
            # while idx < ed:
            #     if isinstance(in_shape[idx], int):
            #         out_data.append(in_shape[idx])
            #     else:
            #         break
            #     idx += st

            # # Only return when in_shape is fully static in the range from begin to end.
            # if idx >= ed:
            #     ret = relax.const(out_data, dtype)
            #     if shrink_axis_mask:
            #         ret = relax.op.nn.squeeze(ret)

            #     return ret

        def _transform_mask(stride_dim, ellipsis_mask):
            """Handle mask inputs to create new begin, end, stride and output shape"""
            m_begin = [0] * data_dim
            m_end = [0] * data_dim
            m_stride = [0] * data_dim
            fshape_indices = []
            # Count new axis after ellipsis_mask, consider while applying ellipsis_mask.
            ellipsis_seen = False
            new_axes_after_ellipsis = 0
            for i in range(stride_dim):
                mask = 1 << i
                if ellipsis_seen and (mask & new_axis_mask) != 0:
                    new_axes_after_ellipsis += 1
                if (mask & ellipsis_mask) != 0:
                    ellipsis_seen = True
            if not ellipsis_seen:
                # Used later for extending the stride attributes in the below loop.
                ellipsis_mask |= 1 << stride_dim
                stride_dim += 1
            final_index = 0
            for index in range(stride_dim):
                mask = 1 << index
                if mask & ellipsis_mask:
                    # Identify the end index for applying ellipsis_mask
                    to_index = min(
                        ((data_dim - (stride_dim - index)) + 1 + new_axes_after_ellipsis), data_dim
                    )
                    for i in range(final_index, to_index):
                        m_begin[final_index] = 0
                        m_end[final_index] = data_shape[final_index]
                        m_stride[final_index] = 1
                        fshape_indices.append(final_index)
                        final_index += 1
                elif mask & new_axis_mask:
                    fshape_indices.append(-1)
                elif not mask & new_axis_mask:
                    if final_index == len(m_begin):
                        break
                    if mask & begin_mask:
                        m_begin[final_index] = -1 if stride[index] < 0 else 0
                    elif begin[index]:
                        m_begin[final_index] = begin[index]
                    if mask & end_mask:
                        m_end[final_index] = (
                            -(data_shape[final_index] + 1)
                            if stride[index] < 0
                            else data_shape[final_index]
                        )
                    elif end[index]:
                        m_end[final_index] = end[index]
                    m_stride[final_index] = stride[index]
                    if mask & shrink_axis_mask:
                        # Tensorflow make axis with shrink_axis_mask as dimension 1
                        m_begin[final_index] = (
                            data_shape[final_index] + begin[index]
                            if begin[index] < 0
                            else begin[index]
                        )
                        m_end[final_index] = m_begin[final_index] + 1
                        m_stride[final_index] = 1
                        fshape_indices.append(-2)
                    else:
                        fshape_indices.append(final_index)

                    final_index += 1
            return m_begin, m_end, m_stride, fshape_indices

        fshape_indices = None
        if begin_mask or end_mask or ellipsis_mask or new_axis_mask or shrink_axis_mask:
            begin, end, stride, fshape_indices = _transform_mask(stride_dim, ellipsis_mask)
        axes = list(range(len(begin)))

        # If input is a constant tensor, we can directly extract it.
        if isinstance(inputs[0], relax.Constant):
            data_np = inputs[0].data.numpy()
            assert len(begin) == len(end) == len(stride)
            slices = [slice(None)] * len(inputs[0].struct_info.shape)
            for idx in range(len(axes)):
                slices[axes[idx]] = slice(begin[idx], end[idx], stride[idx])
            out = data_np[tuple(slices)]
            out_shape = out.shape
            squeeze_func = np.squeeze
            reshape_func = np.reshape
        else:
            out = relax.op.strided_slice(inputs[0], axes=axes, begin=begin, end=end, strides=stride)
            out = bb.normalize(out)
            out_shape = out.struct_info.shape.values
            squeeze_func = relax.op.squeeze
            reshape_func = relax.op.reshape

        if not fshape_indices:
            fshape_indices = range(len(out_shape))

        # Create final output shape.
        final_output = []
        for gather_index in fshape_indices:
            if gather_index == -1:
                final_output.append(1)
            elif gather_index == -2:
                pass
            else:
                final_output.append(out_shape[gather_index])

        if not final_output:
            if not shrink_axis_mask:
                ret = out
            else:
                final_shape = []
                for dim in out_shape:
                    if dim != 1:
                        final_shape.append(dim)
                if len(final_shape) == 0:
                    ret = squeeze_func(out)
                else:
                    # We need reshape to handle dynamic shape.
                    ret = reshape_func(out, tuple(final_shape))
        else:
            ret = reshape_func(out, tuple(final_output))
        return ret

    return _impl


def _pad(name):
    def _impl(inputs, attr, params, bb):
        try:
            padlist = _get_param(params, inputs[1])
        except (IndexError, KeyError, AttributeError):
            padlist = inputs[1]

        assert not isinstance(padlist, relax.Expr), "not supported yet."
        paddings = [item for sublist in padlist for item in sublist]

        attr["pad_width"] = paddings
        attr["pad_value"] = 0
        new_inputs = [inputs[0]]
        if name == "PadV2":
            try:
                attr["pad_value"] = _get_num_param(params, inputs[2])
            except (IndexError, KeyError, AttributeError):
                attr["pad_value"] = inputs[2]
        elif name == "MirrorPad":
            attr["pad_mode"] = attr["mode"].decode("utf-8")
        return AttrCvt(op_name="pad", ignores=["Tpaddings", "mode"])(new_inputs, attr)

    return _impl


def _transpose():
    def _impl(inputs, attr, params, bb):
        # If perm is not specified, axes is left empty,
        # otherwise its value is get from params
        axes = _get_list_param(params, inputs[1], bb)
        return relax.op.permute_dims(inputs[0], axes=axes)

    return _impl


def _where():
    def _impl(inputs, attr, params, bb):
        if len(inputs) == 1:
            return AttrCvt(op_name="argwhere")(inputs, attr)
        cond_shape = inputs[0].struct_info.shape.values
        x_shape = inputs[1].struct_info.shape.values
        # Due to difference in broadcast behavior between Select and SelectV2,
        # we adjust condition dimension with expand_dim and then broadcast.
        if len(cond_shape) == 1 and cond_shape[0] == x_shape[0]:
            for _ in range(len(x_shape) - 1):
                inputs[0] = relax.op.nn.expand_dims(inputs[0], axis=-1)
            broadcast_cond = relax.op.nn.broadcast_to(inputs[0], x_shape)
            inputs[0] = relax.op.nn.cast(broadcast_cond, "bool")
        return AttrCvt(op_name="where")(inputs, attr)

    return _impl


def _where_v2():
    def _impl(inputs, attr, params, bb):
        if len(inputs) == 1:
            return AttrCvt(op_name="nonzero")(inputs, attr)
        return AttrCvt(op_name="where")(inputs, attr)

    return _impl


def _reverse_v2():
    def _impl(inputs, attr, params, bb):
        axis = _get_num_param(params, inputs[1])
        return AttrCvt(op_name="flip", ignores=["Tidx"], extras={"axis": int(axis)})(
            [inputs[0]], attr
        )

    return _impl


def _range():
    def _impl(inputs, attr, params, bb):
        try:
            start = _get_param(params, inputs[0])[0]
        except (IndexError, KeyError, AttributeError):
            try:
                start = _infer_value(inputs[1], params, bb).numpy().tolist()
                start = start if not isinstance(start, list) else start[0]
            except Exception:
                # Symbolic start
                start = inputs[0]

        try:
            limit = (
                _get_param(params, inputs[1])[0]
                if hasattr(inputs[1], "name_hint") or isinstance(inputs[1], relax.Constant)
                else params.pop("Rank").numpy()[0]
            )
        except (IndexError, KeyError, AttributeError):
            try:
                limit = _infer_value(inputs[1], params, bb).numpy().tolist()
                limit = limit if not isinstance(limit, list) else limit[0]
            except Exception:
                limit = inputs[1]

        try:
            delta = _get_param(params, inputs[2])[0]
        except (IndexError, KeyError, AttributeError):
            try:
                delta = _infer_value(inputs[2], params, bb).numpy().tolist()
                delta = delta if not isinstance(delta, list) else delta[0]
            except Exception:
                # Symbolic delta
                delta = inputs[2]

        # if all attributes are constant, evalute the range function and return relay.const
        dtype = attr["Tidx"].name if "Tidx" in attr else str(start.dtype)
        if all(
            [
                isinstance(start, (np.int32, np.int64, int, np.float32, np.float64, float)),
                isinstance(limit, (np.int32, np.int64, int, np.float32, np.float64, float)),
                isinstance(delta, (np.int32, np.int64, int, np.float32, np.float64, float)),
            ]
        ):
            return relax.const(list(range(int(start), int(limit), int(delta))), dtype=dtype)

        if isinstance(start, (np.int32, np.int64, int, np.float32, np.float64, float)):
            start = relax.const(start, dtype=dtype)
        if isinstance(limit, (np.int32, np.int64, int, np.float32, np.float64, float)):
            limit = relax.const(limit, dtype=dtype)
        if isinstance(delta, (np.int32, np.int64, int, np.float32, np.float64, float)):
            delta = relax.const(delta, dtype=dtype)

        return AttrCvt(
            op_name="arange",
            ignores=["Tidx", "_class"],
            extras={"start": start, "stop": limit, "step": delta, "dtype": dtype},
        )([], attr)

    return _impl


def _mean():
    def _impl(inputs, attr, params, bb):
        axis = _get_tuple_param(params, inputs[1])
        return AttrCvt(
            op_name="mean",
            ignores=["Tdim", "Tidx"],
            transforms={"keep_dims": "keepdims"},
            extras={"axis": axis},
        )([inputs[0]], attr)

    return _impl


def _broadcast(name):
    def _impl(inputs, attr, params, bb):
        return AttrCvt(op_name=name, ignores=["name", "incompatible_shape_error", "Tidx"])(
            inputs, attr
        )

    return _impl


def _split(has_size_vector):
    # TF documentation https://www.tensorflow.org/api_docs/python/tf/split
    def _impl(inputs, attr, params, bb):
        try:
            # order and number of inputs are different:
            # if has_size_vector:
            #     https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/split-v
            # else:
            #     https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/split

            # in addition, `axis` and `num_or_size_splits` can be tensors in TensorFlow,
            # we can only support constants
            if has_size_vector:
                input_node_index = 0
                input_axis_index = 2
                size_splits = _get_param(params, inputs[1])
                section_beginnings = np.cumsum(size_splits)[:-1]
                indices_or_sections = tuple(section_beginnings)
            else:
                input_node_index = 1
                input_axis_index = 0
                indices_or_sections = attr["num_split"]
            input_node = inputs[input_node_index]
            axis_input_value = _get_num_param(params, inputs[input_axis_index])
        except (IndexError, KeyError, AttributeError):
            raise TypeError(
                "Unsupported argument for split: `axis` and `num_or_size_splits` "
                "should be constants"
            )
        output = relax.op.split(
            input_node, indices_or_sections=indices_or_sections, axis=int(axis_input_value)
        )
        if isinstance(bb.normalize(output).struct_info, relax.TensorStructInfo):
            return input_node
        return output

    return _impl


def _unpack():
    def _impl(inputs, attr, params, bb):
        input_node = inputs[0]
        axis = attr["axis"]
        input_shape = input_node.struct_info.shape.values
        axis_length = input_shape[axis]
        if axis_length < 0:
            raise TypeError("Unstack with unknown axis length")
        splitted = relax.op.split(input_node, indices_or_sections=axis_length, axis=axis)
        op = bb.normalize(splitted)

        if isinstance(op.struct_info, relax.TensorStructInfo):
            return relax.op.squeeze(input_node, axis=axis)
        return relax.Tuple([relax.op.squeeze(split_item, axis=axis) for split_item in op])

    return _impl


def _softmax():
    def _impl(inputs, attr, params, bb):
        return AttrCvt(op_name="softmax", transforms={"axis": ("axis", 1)})([inputs[0]], attr)

    return _impl


def _topk():
    def _impl(inputs, attr, params, bb):
        k_input = inputs.pop(1)
        try:
            k = int(_get_num_param(params, k_input))
        except (IndexError, KeyError, AttributeError):
            try:
                k = int(_infer_value(k_input, params, bb).numpy().tolist())
            except Exception:
                k = k_input
        if isinstance(k, int):
            if k < 1:
                raise tvm.error.OpAttributeInvalid(
                    "Attribute k must be positive in operator TopKV2"
                )
            k = relax.const(k)
        if attr["sorted"] is False:
            raise tvm.error.OpAttributeUnImplemented(
                "Attribute sorted=False is not supported in operator TopKV2"
            )
        return AttrCvt(
            op_name="topk",
            ignores=["sorted", "Tk", "index_type"],
            extras={"k": k, "is_ascend": False, "dtype": attr["index_type"].name},
        )([inputs[0]], attr)

    return _impl


def _floordiv():
    def _impl(inputs, attr, params, bb):
        assert len(inputs) == 2
        if all(isinstance(inp, relax.Constant) for inp in inputs):
            output = np.floor_divide(inputs[0].data.numpy(), inputs[1].data.numpy())
            return relax.const(output, inputs[0].struct_info.dtype)

        return AttrCvt("floor_divide")(inputs, attr)

    return _impl


def _logical(name):
    def _impl(inputs, attr, params, bb):
        return AttrCvt(op_name=name)(inputs, attr)

    return _impl


def _space_to_batch_nd():
    def _impl(inputs, attr, params, bb):
        block_shape = _get_list_param(params, inputs[1], bb)

        paddings = _get_list_param(params, inputs[2], bb)
        paddings = np.squeeze(paddings)
        if len(paddings.shape) == 1:
            paddings = np.expand_dims(paddings, axis=0)
        paddings = paddings.tolist()

        attr["block_shape"] = block_shape
        attr["paddings"] = paddings
        out = AttrCvt("space_to_batch_nd", ignores=["Tblock_shape", "Tpaddings"])([inputs[0]], attr)

        return out

    return _impl


def _space_to_depth():
    def _impl(inputs, attr, params, bb):
        block_size = int(attr["block_size"])
        layout = attr["data_format"].decode("utf-8")
        return relax.op.nn.space_to_depth(inputs[0], block_size, layout)

    return _impl


def _batch_to_space_nd():
    def _impl(inputs, attr, params, bb):
        block_shape = _get_list_param(params, inputs[1], bb)

        crops = _get_list_param(params, inputs[2], bb)
        crops = np.squeeze(crops)
        if len(crops.shape) == 1:
            crops = np.expand_dims(crops, axis=0)
        crops = crops.tolist()

        attr["block_shape"] = block_shape
        attr["crops"] = crops
        out = AttrCvt("batch_to_space_nd", ignores=["Tblock_shape", "Tcrops"])([inputs[0]], attr)

        return out

    return _impl


def _prod():
    def _impl(inputs, attr, params, bb):
        axis = _get_num_param(params, inputs[1])
        keepdims = attr["keep_dims"]
        inp0 = inputs[0]
        if isinstance(inp0, relax.Constant):
            inp0 = inp0.data.numpy()
            out = np.prod(inp0, axis, dtype=str(inp0.dtype), keepdims=keepdims)
            return relax.const(out)
        return relax.op.statistical.prod(inputs[0], int(axis), keepdims=keepdims)

    return _impl


def _squared_difference():
    def _impl(inputs, attr, params, bb):
        difference = relax.op.subtract(inputs[0], inputs[1])
        return relax.op.multiply(difference, difference)

    return _impl


def _size():
    def _impl(inputs, attr, params, bb):
        new_attr = attr
        new_attr["out_type"] = attr["out_type"].name
        return AttrCvt("ndarray_size", transforms={"out_type": "dtype"})(inputs, new_attr)

    return _impl


def _ctc_greedy_decoder():
    def _impl(inputs, attr, params, bb):
        from tvm.compass.relax import op as compass_op  # pylint: disable=import-outside-toplevel

        data = inputs[0]
        seq_len = inputs[1]
        merge_repeated = attr["merge_repeated"]

        # for compass only support [batch_size, max_time, num_classes]
        transpose = relax.op.permute_dims(data, [1, 0, 2])
        return compass_op.ctc_greedy_decoder(transpose, seq_len, merge_repeated)

    return _impl


def _matrix_band_part():
    def _impl(inputs, attr, params, bb):
        if not all(isinstance(inp, relax.Constant) for inp in inputs):
            msg = "Only supports inputs of matrix_band_part as constant value"
            raise tvm.error.OpAttributeInvalid(msg)
        inp = inputs[0].data.numpy()
        lower = int(inputs[1].data.numpy())
        upper = int(inputs[2].data.numpy())
        output = tf.linalg.band_part(inp, lower, upper).numpy()
        return relax.const(output, output.dtype)

    return _impl


def _fake_quant_with_min_max_vars():
    def _impl(inputs, attr, params, bb):
        from tvm.compass.relax import op as compass_op  # pylint: disable=import-outside-toplevel

        data = inputs[0]
        minimum = inputs[1].data.numpy()
        maximum = inputs[2].data.numpy()
        narrow_range = attr["narrow_range"]
        num_bits = attr["num_bits"]

        return compass_op.fake_quant_with_min_max_vars(
            data, float(minimum), float(maximum), narrow_range, num_bits
        )

    return _impl


def _leakyrelu():
    def _impl(inputs, attr, params, bb):
        return relax.op.nn.leakyrelu(inputs[0], attr["alpha"])

    return _impl


# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)
_convert_map = {
    "Add": _elemwise("add"),
    "AddV2": _elemwise("add"),
    "All": _reduce("all"),
    "Any": _reduce("any"),
    "ArgMax": _argx(relax.op.argmax, "argmax"),
    "ArgMin": _argx(relax.op.argmin, "argmin"),
    "Assert": _assert(),
    "AvgPool": _pooling("avg_pool"),
    "BatchMatMulV2": _batch_matmul(),
    "BatchToSpaceND": _batch_to_space_nd(),
    "BiasAdd": _bias_add(),
    "Cast": _cast(),
    "ConcatV2": _concatV2(),
    "Conv2D": _conv("conv"),
    "Conv2DBackpropInput": _conv("conv_transpose"),
    "CropAndResize": _crop_and_resize(),
    "DepthwiseConv2dNative": _conv("depthwise"),
    "Equal": _broadcast("equal"),
    "Exp": AttrCvt("exp"),
    "ExpandDims": _expand_dims(),
    "Fill": _fill(),
    "FloorDiv": _floordiv(),
    "FusedBatchNorm": _fused_batch_norm(),
    "FusedBatchNormV3": _fused_batch_norm(),
    "Gather": _gather(),
    "GatherV2": _gather(),
    "Greater": _broadcast("greater"),
    "GreaterEqual": _broadcast("greater_equal"),
    "Identity": _identity(),
    "IdentityN": _identityn(),
    "LeakyRelu": _leakyrelu(),
    "Less": _broadcast("less"),
    "LessEqual": _broadcast("less_equal"),
    "LogicalAnd": _logical("logical_and"),
    "LogicalNot": _logical("logical_not"),
    "LogicalOr": _logical("logical_or"),
    "MatMul": _matmul(),
    "Max": _reduce("max"),
    "Min": _reduce("min"),
    "Maximum": _elemwise("maximum"),
    "MaxPool": _pooling("max_pool"),
    "Mean": _mean(),
    "Minimum": _elemwise("minimum"),
    "MirrorPad": _pad("MirrorPad"),
    "Mul": _elemwise("multiply"),
    "NotEqual": _broadcast("not_equal"),
    "NonMaxSuppressionV2": _nms(),
    "NonMaxSuppressionV3": _nms(),
    "OneHot": _one_hot(),
    "Pack": _pack(),
    "Pad": _pad("Pad"),
    "PadV2": _pad("PadV2"),
    "Pow": _elemwise("power"),
    "Prod": _prod(),
    "Range": _range(),
    "RealDiv": _elemwise("divide"),
    "Relu": AttrCvt("relu"),
    "Relu6": _relu6(),
    "Reshape": _reshape(),
    "ResizeBilinear": _resize("linear"),
    "ResizeNearestNeighbor": _resize("nearest_neighbor"),
    "ReverseV2": _reverse_v2(),
    "Round": AttrCvt("round"),
    "Rsqrt": _rsqrt(),
    "Select": _where(),
    "Shape": _shape(),
    "Sigmoid": AttrCvt("sigmoid"),
    "Size": _size(),
    "Slice": _slice(),
    "Softmax": _softmax(),
    "SpaceToBatchND": _space_to_batch_nd(),
    "SpaceToDepth": _space_to_depth(),
    "SparseToDense": _sparse_to_dense(),
    "Split": _split(False),
    "SplitV": _split(True),
    "Square": _square(),
    "SquaredDifference": _squared_difference(),
    "Squeeze": _squeeze(),
    "StopGradient": _identity(),
    "StridedSlice": _stridedSlice(),
    "Sub": _elemwise("subtract"),
    "Sum": _sum(),
    "Tanh": AttrCvt("tanh"),
    "TensorArrayGatherV3": _tensor_array_gather(),
    "TensorArrayReadV3": _tensor_array_read(),
    "TensorArrayScatterV3": _tensor_array_scatter(),
    "TensorArraySizeV3": _tensor_array_size(),
    "TensorArrayV3": _tensor_array(),
    "TensorArrayWriteV3": _tensor_array_write(),
    "Tile": _tile(),
    "TopKV2": _topk(),
    "Transpose": _transpose(),
    "Unpack": _unpack(),
    "Where": _where_v2(),
    "ZerosLike": AttrCvt("zeros_like"),
    "CTCGreedyDecoder": _ctc_greedy_decoder(),
    "MatrixBandPart": _matrix_band_part(),
    "FakeQuantWithMinMaxVars": _fake_quant_with_min_max_vars(),
}
