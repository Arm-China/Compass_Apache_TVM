# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
"""Operators extended by AIPU Compass."""
import numpy as np
import tvm
from tvm import relay, ir
from tvm.relay.transform.infer_layout_utils import InferCorrectLayoutOutput
from . import _make
from ...nn.utils import get_pad_tuple2d
from ....op import op as reg


def _tile_anchor(
    grid_height,
    grid_width,
    scales_grid,
    aspect_ratios_grid,
    anchor_stride_,
    anchor_offset_,
    base_anchor_size,
):
    """
    get the fixed anchor with the given anchor_scale and anchor_aspect_ratio.
    """

    scales_grid = np.reshape(scales_grid, [-1])
    aspect_ratios_grid = np.reshape(aspect_ratios_grid, [-1])
    ratio_sqrts = np.sqrt(aspect_ratios_grid)

    heights = scales_grid / ratio_sqrts * base_anchor_size[0]
    widths = scales_grid * ratio_sqrts * base_anchor_size[1]

    y_centers = np.array(range(grid_height), dtype=np.float)
    y_centers = y_centers * anchor_stride_[0] + anchor_offset_[0]
    x_centers = np.array(range(grid_width), dtype=np.float)
    x_centers = x_centers * anchor_stride_[1] + anchor_offset_[1]
    x_centers, y_centers = np.meshgrid(x_centers, y_centers)

    widths_grid, x_centers_grid = np.meshgrid(widths, x_centers)
    heights_grid, y_centers_grid = np.meshgrid(heights, y_centers)

    bbox_centers = np.stack(
        [y_centers_grid[:, :, np.newaxis], x_centers_grid[:, :, np.newaxis]], axis=3
    )
    bbox_sizes = np.stack([heights_grid[:, :, np.newaxis], widths_grid[:, :, np.newaxis]], axis=3)
    bbox_centers = np.reshape(bbox_centers, [-1, 2])
    bbox_sizes = np.reshape(bbox_sizes, [-1, 2])

    bbox_corners = np.concatenate(
        [bbox_centers - 0.5 * bbox_sizes, bbox_centers + 0.5 * bbox_sizes], 1
    )

    y_min, x_min, y_max, x_max = np.split(bbox_corners, 4, axis=1)

    win_y_min = 0
    win_x_min = 0
    win_y_max = 1.0
    win_x_max = 1.0
    y_min = np.clip(y_min, win_y_min, win_y_max)
    y_max = np.clip(y_max, win_y_min, win_y_max)
    x_min = np.clip(x_min, win_x_min, win_x_max)
    x_max = np.clip(x_max, win_x_min, win_x_max)

    bboxes = np.concatenate([y_min, x_min, y_max, x_max], 1)
    areas = np.squeeze((y_max - y_min) * (x_max - x_min))
    bboxes = bboxes[areas > 0]

    ycenter_a = (bboxes[:, 0] + bboxes[:, 2]) / 2
    xcenter_a = (bboxes[:, 1] + bboxes[:, 3]) / 2
    height_a = bboxes[:, 2] - bboxes[:, 0]
    width_a = bboxes[:, 3] - bboxes[:, 1]

    return ycenter_a, xcenter_a, height_a, width_a


def gen_anchor_box(feature_map_):
    """Generate anchor box with feature map."""
    min_scale = 0.2
    max_scale = 0.950000
    num_layers = 6
    anchor_scales_ = np.linspace(min_scale, max_scale, num_layers).tolist()

    append_anchor_scales = anchor_scales_ + [1.0]
    inter_scales = [
        np.sqrt(i * j) for i, j in zip(append_anchor_scales[:-1], append_anchor_scales[1:])
    ]
    firstbox_scale = [10.0, 5.0, 5.0]
    anchor_aspect_ratios_ = [1.0, 2.0, 0.5, 3.0, 0.333333]
    idx = 0
    total_ycenter_a = None
    total_xcenter_a = None
    total_ha = None
    total_wa = None
    for scale, feat_map in zip(anchor_scales_, feature_map_):
        aspect_ratios = []
        scales = []

        if idx == 0:
            scales = 1.0 / np.reshape(firstbox_scale, -1)
            aspect_ratios = [1.0, 2.0, 0.5]
        else:
            for aspect_ratio in anchor_aspect_ratios_:
                scales.append(scale)
                aspect_ratios.append(aspect_ratio)
            scales.append(inter_scales[idx])
            aspect_ratios.append(1.0)
        idx += 1

        feat_h, feat_w = feat_map[0], feat_map[1]
        anchor_stride_ = [1.0 / feat_h, 1.0 / feat_w]
        anchor_offset_ = [0.5 * a_s for a_s in anchor_stride_]
        base_anchor_size = [1.0, 1.0]

        ycenter_a, xcenter_a, height_a, width_a = _tile_anchor(
            feat_h,
            feat_w,
            scales,
            aspect_ratios,
            anchor_stride_,
            anchor_offset_,
            base_anchor_size,
        )

        if total_ycenter_a is None:
            total_ycenter_a = ycenter_a
            total_xcenter_a = xcenter_a
            total_ha = height_a
            total_wa = width_a
        else:
            total_ycenter_a = np.concatenate([total_ycenter_a, ycenter_a])
            total_xcenter_a = np.concatenate([total_xcenter_a, xcenter_a])
            total_ha = np.concatenate([total_ha, height_a])
            total_wa = np.concatenate([total_wa, width_a])

    return total_ycenter_a, total_xcenter_a, total_ha, total_wa


@tvm.register_func("relay.op.contrib.aipu_compass.gruv3_rel")
def _gruv3_rel(types, out_sequences, activations):
    """GRUv3 infer type function."""
    inp, _, _, bias = types[:4]
    batch, time_steps, _ = inp.shape
    cell_size = bias.shape[0] // 3
    dtype = inp.dtype
    if activations != "SIGMOID,TANH":
        return "activations only support 'SIGMOID,TANH'"
    if out_sequences not in ["H", "Hn", "H,Hn"]:
        return "out_sequences only support 'H', 'Hn', 'H,Hn'"
    if out_sequences == "H":
        ret_type = relay.TensorType([batch, time_steps, cell_size], dtype)
        return [ret_type]
    if out_sequences == "Hn":
        ret_type = relay.TensorType([batch, cell_size], dtype)
        return [ret_type]
    ret_type0 = relay.TensorType([batch, time_steps, cell_size], dtype)
    ret_type1 = relay.TensorType([batch, cell_size], dtype)
    return [ret_type0, ret_type1]


def gruv3(seq, init_state, kernels, biases, out_sequences, activations="SIGMOID,TANH"):
    """Make an AIPU Compass GRUv3 operator.

    Parameters
    ----------
    seq : relay.Expr
        The input of GRUv3.

    init_state : relay.Expr
        The init_state of GRUv3.

    kernels : relay.Expr
        The kernels of GRUv3.

    biases : relay.Expr
        The biases of GRUv3.

    out_sequences : string
        string indicate the out_sequences, only support "H", "Hn", "H,Hn".

    activations : string
        string indicate the activations of GRUv3, only support "SIGMOID,TANH".

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.gruv3(seq, init_state, kernels, biases, out_sequences, activations)


def ctc_greedy_decoder(data, seq_len, merge_repeated=True):
    """Make a TensorFlow CTCGreedyDecoder operator.

    Parameters
    ----------
    data : relay.Expr
        The input tensor.

    seq_len : relay.Expr
        1-D int32 vector containing sequence lengths.

    merge_repeated : bool
        Whether merge repeated classes in output.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.ctc_greedy_decoder(data, seq_len, merge_repeated)


def matrix_band_part(data, num_lower, num_upper):
    """Make a TensorFlow MatrixBandPart operator.

    Parameters
    ----------
    data : relay.Expr
        The input matrix

    num_lower : int
        The init_state of MatrixBandPart

    num_upper : int
        The kernels of MatrixBandPart

    Returns
    -------
    result : relay.Expr
        The computed result.
    """

    return _make.matrix_band_part(data, num_lower, num_upper)


def fake_quant_with_min_max_vars(data, minimum, maximum, narrow_range=False, num_bits=8):
    """Make a TensorFlow FakeQuantWithMinMaxVars operator.
    data : The input tensor.

    minimum : float
        The clip minimum for input.

    maximum : float
        The clip maximum for input.

    narrow_range : bool
        Quant to [0,2^num_bits - 1] when false and [1,2^num_bits - 1] when true.

    num_bits : int
        The bitwidth of the quantization.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """

    return _make.fake_quant_with_min_max_vars(data, minimum, maximum, narrow_range, num_bits)


def deformable_conv2d_v2(
    data,
    offset,
    weight,
    mask,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    deformable_groups=1,
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCHW",
    kernel_layout="OIHW",
    out_layout="",
    out_dtype="",
):
    r"""Deformable 2d v2 convolution.

    The deformable convolution v2 operation is described in https://arxiv.org/abs/1811.11168

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    offset : tvm.relay.Expr
        The offset expressions.

    weight : tvm.relay.Expr
        The weight expressions.

    mask : tvm.relay.Expr
        The mask expressions.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    deformable_groups : int, optional
        Number of deformable groups.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the weight.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.

    """
    # convert 2-way padding to 4-way padding
    padding = get_pad_tuple2d(padding)
    return _make.deformable_conv2d_v2(
        data,
        offset,
        weight,
        mask,
        strides,
        padding,
        dilation,
        deformable_groups,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


# pylint: disable=unused-argument
@reg.register_convert_op_layout("contrib.aipu_compass.deformable_conv2d_v2")
def convert_deformable_conv2d_v2(attrs, inputs, tinfos, desired_layouts):
    """Convert Layout pass registration for deformable conv2d op.
    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layouts : list of layout strings
        List of layouts defining our desired
        layout for the data and kernel inputs respectively.
    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """
    data, offset, weight, mask = inputs
    new_attrs = dict(attrs)
    for attr in new_attrs:
        if isinstance(new_attrs[attr], ir.Array):
            new_attrs[attr] = list(new_attrs[attr])
        elif isinstance(new_attrs[attr], tvm.tir.expr.IntImm):
            new_attrs[attr] = new_attrs[attr].value

    # First check if there is a LayoutConfig scope, and if so, whether
    # it indicates we should ignore this layer or not.
    layout_config = relay.transform.LayoutConfig.current
    if layout_config is not None:
        skip_layer = layout_config.check_skip()
        if skip_layer:
            return deformable_conv2d_v2(data, offset, weight, mask, **new_attrs)

    # Prepare new layout.
    assert len(desired_layouts) == 2, "A desired layout is expected for data and kernel"
    desired_data_layout, desired_kernel_layout = map(str, desired_layouts)
    assert desired_data_layout != "default", "Data layout cannot be default"
    new_attrs["data_layout"] = desired_data_layout

    if desired_kernel_layout != "default":
        new_attrs["kernel_layout"] = desired_kernel_layout
        return deformable_conv2d_v2(data, offset, weight, mask, **new_attrs)

    # Handle default kernel layouts
    if desired_data_layout == "NCHW":
        new_attrs["kernel_layout"] = "OIHW"
    elif desired_data_layout == "NHWC":
        new_attrs["kernel_layout"] = "HWIO"
    else:
        raise ValueError("Layout %s is not yet supported." % desired_data_layout)

    return deformable_conv2d_v2(data, offset, weight, mask, **new_attrs)


# pylint: disable=unused-argument
@reg.register_convert_op_layout("nn.adaptive_avg_pool2d")
def convert_nn_adaptive_avg_pool2d(attrs, inputs, tinfos, desired_layouts):
    """Convert Layout pass registration for nn adaptive_avg_pool2d op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current resize op
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layouts : list of layout strings
        List of layouts defining our desired
        layout for the data input.
    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """
    new_attrs = dict(attrs)
    assert len(desired_layouts) == 1, "Only one desired layout is expected"
    desired_layout = str(desired_layouts[0])
    assert desired_layout != "default", "Layout cannot be default"
    new_attrs["layout"] = desired_layout
    new_attrs["out_layout"] = desired_layout
    return relay.nn.adaptive_avg_pool2d(*inputs, **new_attrs)


@reg.register_infer_correct_layout("nn.lrn")
def infer_correct_layout_lrn(attrs, new_in_layouts, old_in_layouts, old_in_types):
    """
    Parameters
    ----------
    attrs : tvm.ir.Attrs
        The attribute of current op.
    new_in_layouts : list of layouts
        The layouts of input arguments after alter_op_layout.
        This can be undefined, which means we call this function before alternating
        any operators.
    old_in_layouts : list of layouts
        The layouts of input arguments before alter_op_layout.
    old_in_types : list of types
        The types of old input arguments.
    Returns
    -------
    result : InferCorrectLayoutOutput
        Inferred layouts and updated attributes stored in InferCorrectLayoutOutput above.
    """
    ret = tvm.tir.layout("")
    new_attrs = dict()
    for k in attrs.keys():
        # Here runtime.String pass on as kTVMObjectHandle to c++, but need kTVMStr
        # So change runtime.String to python string
        new_attrs[str(k)] = attrs[k]

    if new_in_layouts and old_in_layouts:
        assert len(new_in_layouts) >= 1 and len(old_in_layouts) >= 1
        lrn_dim = old_in_layouts[0][new_attrs["axis"]]
        new_index = new_in_layouts[0].index_of(lrn_dim)
        new_attrs["axis"] = new_index
        ret = new_in_layouts[0]
    elif old_in_layouts:
        ret = old_in_layouts[0]

    return InferCorrectLayoutOutput([ret], [ret], ir.make_node("relay.attrs.LRNAttrs", **new_attrs))


def channel_shuffle(data, group, splits):
    """Make a channel shuffle operator.
    data : The input tensor.

    group : int
        The group number of the input tensor.

    splits : int
        The number of output tensors.

    Returns
    -------
    result : relay.Expr
    """
    return _make.channel_shuffle(data, group, splits)


def custom_op(data, out_type):
    """Make a customized fixed type relation operator with the given inputs.

    Parameters
    ----------
    data : Union[relay.Expr, List[relay.Expr], Tuple[relay.Expr]]
        The inputs of this customized operator.

    out_type : ir.Type
        The output type of the customized operator.
    """
    if isinstance(data, (list, tuple, ir.Array)):
        if len(data) == 1:
            data = data[0]
        else:
            data = relay.Tuple(data)
    return _make.custom_op(data, out_type)


def _detection_output_type_rel(arg_types, attrs):  # pylint: disable=unused-argument
    assert len(arg_types) == 3
    box_encodings_shape = [int(dim) for dim in arg_types[1].shape]
    batch_size, crop_box, max_class, _ = box_encodings_shape
    max_detection = crop_box * max_class

    scores = relay.TensorType([batch_size, max_detection])
    boxes = relay.TensorType([batch_size, max_detection, 4])
    box_num_perclass = relay.TensorType([batch_size, max_class])
    class_label = relay.TensorType([batch_size, max_class])
    total_class = relay.TensorType([batch_size, 1])
    return relay.TupleType([scores, boxes, box_num_perclass, class_label, total_class])


def detection_output(
    class_conf,
    box_encodings,
    crop_proposals,
    image_width=300,
    image_height=300,
    score_threshold=0.7,
    variance=None,
):
    """Make an AIPU Compass DetectionOutput operator.

    Parameters
    ----------
    class_conf : relay.Expr
        The input classes.

    box_encodings : relay.Expr
        The input boxes.

    crop_proposals : relay.Expr
        The input proposals.

    image_width : int
        The width of image.

    image_height : int
        The height of image.

    score_threshold : float
        The threshold to choose box.

    variance : list of float

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    params = [class_conf, box_encodings, crop_proposals]
    attrs = {
        "image_width": image_width,
        "image_height": image_height,
        "score_threshold": score_threshold,
        "variance": variance,
    }
    attrs_node = ir.make_node("DictAttrs", **attrs)
    new_call = relay.Call(relay.op.get("contrib.aipu_compass.detection_output"), params, attrs_node)
    out = relay.TupleWrapper(new_call, 5)
    out = relay.Tuple(list(out))
    return out


def _nms_type_rel(arg_types, attrs):
    assert len(arg_types) == 4
    proposal_boxes, box_num_per_class, _, _ = arg_types
    batch = int(box_num_per_class.shape[0])
    max_class_num = int(box_num_per_class.shape[1])
    max_nms_box_num = int(proposal_boxes.shape[1])
    if "max_output_size" in attrs.keys():
        max_nms_box_num = attrs["max_output_size"]

    nms_boxes = relay.TensorType([batch, max_nms_box_num, 4])
    nms_box_num_per_class = relay.TensorType([batch, max_class_num])
    nms_scores = relay.TensorType([batch, max_nms_box_num])
    keep = relay.TensorType([batch, max_nms_box_num])
    return relay.TupleType([nms_boxes, nms_box_num_per_class, nms_scores, keep])


def nms(
    proposal_boxes,
    box_num_per_class,
    total_class_num,
    proposal_scores,
    image_width=300,
    image_height=300,
    score_threshold=-float("inf"),
    method="HARD",
    iou_threshold=0.6,
    max_output_size=5000,
    center_point_box=0,
    soft_nms_sigma=0.0,
):
    """Make an AIPU Compass NMS operator.

    Parameters
    ----------
    proposal_boxes : relay.Expr
        The input boxes.

    box_num_per_class : relay.Expr
        The box num of per class.

    total_class_num : relay.Expr
        The total class num.

    proposal_scores : relay.Expr
        The input scores.

    image_width : int
        The width of image.

    image_height : int
        The height of image.

    score_threshold : float
        The threshold to choose box.

    method : string
        The method. Option in ['HARD', 'GAUSSIAN', 'LINEAR'].

    iou_threshold : float
        The threshold in iou compute.

    max_output_size : int
        The max output size.

    center_point_box : int
        The center point box. Option in [0, 1]

    soft_nms_sigma : float

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    params = [proposal_boxes, box_num_per_class, total_class_num, proposal_scores]
    attrs = {
        "method": method,
        "image_width": image_width,
        "image_height": image_height,
        "center_point_box": center_point_box,
        "max_output_size": max_output_size,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "soft_nms_sigma": soft_nms_sigma,
    }
    attrs_node = ir.make_node("DictAttrs", **attrs)
    new_call = relay.Call(relay.op.get("contrib.aipu_compass.nms"), params, attrs_node)
    out = relay.TupleWrapper(new_call, 4)
    out = relay.Tuple(list(out))
    return out


def _decode_box_rel(arg_types, attrs):
    batch = int(arg_types[0].shape[0])
    max_box_num = attrs["max_box_num"]

    selected_box = relay.TensorType([batch, max_box_num, 4])
    box_num_per_class = relay.TensorType([batch, max_box_num])
    class_num = relay.TensorType([batch, 1])
    box_scores = relay.TensorType([batch, max_box_num])
    box_labels = relay.TensorType([batch, max_box_num])
    return relay.TupleType([selected_box, box_num_per_class, class_num, box_scores, box_labels])


def decode_box(
    scores,
    boxes,
    feature_map,
    image_width=300,
    image_height=300,
    max_box_num=5000,
    class_num=90,
    score_threshold=0.5,
):
    """Make an AIPU Compass DecodeBox operator.

    Parameters
    ----------
    scores : relay.Expr
        The input scores.

    boxes : relay.Expr
        The input boxes.

    feature_map : relay.Expr
        The feature_map to generate anchor boxes.

    image_width : int
        The width of image.

    image_height : int
        The height of image.

    max_box_num : int
        The max box num.

    class_num : int
        The class num.

    score_threshold : float
        The threshold to choose box.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    ycenter, xcenter, ha_data, wa_data = gen_anchor_box(feature_map)
    ycenter = relay.const(ycenter, "float32")
    xcenter = relay.const(xcenter, "float32")
    ha_data = relay.const(ha_data, "float32")
    wa_data = relay.const(wa_data, "float32")
    params = [scores, boxes, ycenter, xcenter, ha_data, wa_data]
    attrs = {
        "image_width": image_width,
        "image_height": image_height,
        "max_box_num": max_box_num,
        "class_num": class_num,
        "score_threshold": score_threshold,
    }
    attrs_node = ir.make_node("DictAttrs", **attrs)
    new_call = relay.Call(relay.op.get("contrib.aipu_compass.decode_box"), params, attrs_node)
    out = relay.TupleWrapper(new_call, 5)
    out = relay.Tuple(list(out))
    return out


def register_op(op_name, input_num, rel_func):
    relay.op.op.register(op_name)
    op = relay.op.get(op_name)
    op.set_num_inputs(input_num)
    op.set_attrs_type_key("DictAttrs")
    op.add_type_rel(op_name, rel_func)


register_op("contrib.aipu_compass.detection_output", 3, _detection_output_type_rel)
register_op("contrib.aipu_compass.nms", 4, _nms_type_rel)
register_op("contrib.aipu_compass.decode_box", 6, _decode_box_rel)
