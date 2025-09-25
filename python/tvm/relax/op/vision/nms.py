# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Non-maximum suppression operations."""
from tvm import relax
from tvm.relax import expr
from . import _ffi_api


def all_class_non_max_suppression(
    boxes,
    scores,
    max_output_boxes_per_class=-1,
    iou_threshold=-1.0,
    score_threshold=-1.0,
    output_format="onnx",
):
    """Non-maximum suppression operator for object detection, corresponding to ONNX
    NonMaxSuppression and TensorFlow combined_non_max_suppression.
    NMS is performed for each class separately.

    Parameters
    ----------
    boxes : relax.Expr
        3-D tensor with shape (batch_size, num_boxes, 4)

    scores: relax.Expr
        3-D tensor with shape (batch_size, num_classes, num_boxes)

    max_output_boxes_per_class : int or relax.Expr, optional
        The maxinum number of output selected boxes per class

    iou_threshold : float or relax.Expr, optional
        IoU test threshold

    score_threshold : float or relax.Expr, optional
        Score threshold to filter out low score boxes early

    output_format : string, optional
        "onnx" or "tensorflow". Specify by which frontends the outputs are
        intented to be consumed.

    Returns
    -------
    out : relax.Tuple
        If `output_format` is "onnx", the output is a relax.Tuple of two tensors, the first is
        `indices` of size `(batch_size * num_class* num_boxes , 3)` and the second is a scalar
        tensor `num_total_detection` of shape `(1,)` representing the total number of selected
        boxes. The three values in `indices` encode batch, class, and box indices.
        Rows of `indices` are ordered such that selected boxes from batch 0, class 0 come first,
        in descending of scores, followed by boxes from batch 0, class 1 etc. Out of
        `batch_size * num_class* num_boxes` rows of indices,  only the first `num_total_detection`
        rows are valid.

        If `output_format` is "tensorflow", the output is a relax.Tuple of three tensors, the first
        is `indices` of size `(batch_size, num_class * num_boxes , 2)`, the second is `scores` of
        size `(batch_size, num_class * num_boxes)`, and the third is `num_total_detection` of size
        `(batch_size,)` representing the total number of selected boxes per batch. The two values
        in `indices` encode class and box indices. Of num_class * num_boxes boxes in `indices` at
        batch b, only the first `num_total_detection[b]` entries are valid. The second axis of
        `indices` and `scores` are sorted within each class by box scores, but not across classes.
        So the box indices and scores for the class 0 come first in a sorted order, followed by
        the class 1 etc.
    """
    if not isinstance(max_output_boxes_per_class, expr.Expr):
        max_output_boxes_per_class = relax.const(max_output_boxes_per_class, "int32")
    if not isinstance(iou_threshold, expr.Expr):
        iou_threshold = relax.const(iou_threshold, "float32")
    if not isinstance(score_threshold, expr.Expr):
        score_threshold = relax.const(score_threshold, "float32")

    out = _ffi_api.all_class_non_max_suppression(
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        output_format,
    )
    return out
