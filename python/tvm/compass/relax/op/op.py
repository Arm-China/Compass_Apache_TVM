# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=unused-argument
"""Operators extended by Zhouyi Compass."""
import numpy as np
from tvm import ir, relax
from tvm.tir import IntImm
from . import _ffi_api


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

    y_centers = np.array(range(grid_height), dtype=float)
    y_centers = y_centers * anchor_stride_[0] + anchor_offset_[0]
    x_centers = np.array(range(grid_width), dtype=float)
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


def _infer_sinfo_decode_box(call: relax.Call, context):
    batch = int(call.args[0].struct_info.shape[0])
    max_box_num = call.attrs["max_box_num"]

    selected_box = relax.TensorStructInfo([batch, max_box_num, 4])
    box_num_per_class = relax.TensorStructInfo([batch, max_box_num])
    class_num = relax.TensorStructInfo([batch, 1])
    box_scores = relax.TensorStructInfo([batch, max_box_num])
    box_labels = relax.TensorStructInfo([batch, max_box_num])
    return relax.TupleStructInfo(
        [selected_box, box_num_per_class, class_num, box_scores, box_labels]
    )


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
    """Make a Compass DecodeBox operator.

    Parameters
    ----------
    scores : relax.Expr
        The input scores.

    boxes : relax.Expr
        The input boxes.

    feature_map : relax.Expr
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
    result : relax.Expr
        The computed result.
    """
    ycenter, xcenter, ha_data, wa_data = gen_anchor_box(feature_map)
    ycenter = relax.const(ycenter, "float32")
    xcenter = relax.const(xcenter, "float32")
    ha_data = relax.const(ha_data, "float32")
    wa_data = relax.const(wa_data, "float32")
    params = [scores, boxes, ycenter, xcenter, ha_data, wa_data]
    attrs = {
        "image_width": image_width,
        "image_height": image_height,
        "max_box_num": max_box_num,
        "class_num": class_num,
        "score_threshold": score_threshold,
    }
    attrs_node = ir.make_node("DictAttrs", **attrs)
    return relax.Call(ir.Op.get("relax.decode_box"), params, attrs_node)


def _infer_sinfo_nms(call: relax.Call, context):
    assert len(call.args) == 4
    proposal_boxes, box_num_per_class, _, _ = call.args
    batch = int(box_num_per_class.struct_info.shape[0])
    max_class_num = int(box_num_per_class.struct_info.shape[1])
    max_nms_box_num = int(proposal_boxes.struct_info.shape[1])
    max_nms_box_num = call.attrs.get("max_output_size", max_nms_box_num)

    nms_boxes = relax.TensorStructInfo([batch, max_nms_box_num, 4])
    nms_box_num_per_class = relax.TensorStructInfo([batch, max_class_num])
    nms_scores = relax.TensorStructInfo([batch, max_nms_box_num])
    keep = relax.TensorStructInfo([batch, max_nms_box_num])
    return relax.TupleStructInfo([nms_boxes, nms_box_num_per_class, nms_scores, keep])


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
    """Make a Compass NMS operator.

    Parameters
    ----------
    proposal_boxes : relax.Expr
        The input boxes.

    box_num_per_class : relax.Expr
        The box num of per class.

    total_class_num : relax.Expr
        The total class num.

    proposal_scores : relax.Expr
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
    result : relax.Expr
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
    return relax.Call(ir.Op.get("relax.cps_nms"), params, attrs_node)


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
    result : relax.Expr
        The computed result.
    """
    op = ir.Op.get("relax.fake_quant_with_min_max_vars")
    attrs = {
        "minimum": minimum,
        "maximum": maximum,
        "narrow_range": narrow_range,
        "num_bits": num_bits,
    }
    attr_node = ir.make_node("DictAttrs", **attrs)

    return relax.Call(op, [data], attr_node)


def ctc_greedy_decoder(data, seq_len, merge_repeated=True):
    """Make a TensorFlow CTCGreedyDecoder operator.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    seq_len : relax.Expr
        1-D int32 vector containing sequence lengths.

    merge_repeated : bool
        Whether merge repeated classes in output.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    op = ir.Op.get("relax.ctc_greedy_decoder")
    attr_node = ir.make_node("DictAttrs", merge_repeated=merge_repeated)
    return relax.Call(op, [data, seq_len], attr_node)


def _infer_sinfo_ctc_greedy_decoder(call: relax.Call, context):
    data, seq_len = call.args
    assert data.struct_info.shape[0] == seq_len.struct_info.shape[0], "batch_size should be same."
    out_shape = [data.struct_info.shape[0], 4096, 1, 1]
    return relax.TensorStructInfo(out_shape, "int32")


def requantize(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    axis=-1,
    out_dtype="int8",
):
    r"""Requantized operator.

    The requantize operator converts one quantized tensor representation to
    another quantized tensor representation. For the output tensor, we are
    provided with output scale and zero point. The computation is as follows

    Q_output = zp_output +  (scale_input)/(scale_output) * (Q_input - zp_input)
    (Equals dequantize + quantize.)

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    input_scale: relax.Expr
        The quantization scale for the input tensor.

    input_zero_point: relax.Expr
        The zero point of the input tensor.

    output_scale: relax.Expr
        The quantization scale for the output tensor.

    output_zero_point: relax.Expr
        The zero point of the output tensor.

    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.

    out_dtype : str, optional
        Specifies the output data type.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    op = ir.Op.get("relax.requantize")
    attr_node = ir.make_node("DictAttrs", axis=axis, out_dtype=out_dtype)
    inps = [data, input_scale, input_zero_point, output_scale, output_zero_point]
    return relax.Call(op, inps, attr_node)


def _infer_sinfo_requantize(call: relax.Call, context):
    return relax.TensorStructInfo(call.args[0].struct_info.shape, call.attrs.out_dtype)


def channel_shuffle(data, group, axis, splits):
    """Make a channel shuffle operator.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    group : int
        The group number of the input tensor.

    axis : int
        The dimension along which to shuffle.

    splits : int
        The number of output tensors.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    args = [int(x) if isinstance(x, IntImm) else x for x in (group, axis, splits)]
    return _ffi_api.channel_shuffle(data, *args)


def default_infer_struct_info(call: relax.Call, context):
    """Default infer struct info function to return struct info of inp0."""
    return call.args[0].struct_info


def register_op(op_name, num_inputs=1, infer_func=default_infer_struct_info):
    ir.register_op_attr(op_name, "FInferStructInfo", infer_func)
    ir.register_op_attr(op_name, "FPurity", True)
    op = ir.Op.get(op_name)
    op.set_num_inputs(num_inputs)
    op.set_attrs_type_key("DictAttrs")


register_op("relax.fake_quant_with_min_max_vars")
register_op("relax.ctc_greedy_decoder", 2, _infer_sinfo_ctc_greedy_decoder)
register_op("relax.requantize", 5, _infer_sinfo_requantize)
register_op("relax.decode_box", 6, _infer_sinfo_decode_box)
register_op("relax.cps_nms", 4, _infer_sinfo_nms)
