# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import tvm
import tvm.testing
from tvm import relax, tir
from tvm import TVMError
from tvm.ir import Op, VDevice
from tvm.script import relax as R


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_all_class_non_max_suppression_infer_struct_info():
    bb = relax.BlockBuilder()
    batch_size, num_classes, num_boxes = 10, 8, 5
    boxes = relax.Var("boxes", R.Tensor((batch_size, num_boxes, 4), "int64"))
    scores = relax.Var("scores", R.Tensor((batch_size, num_classes, num_boxes), "float32"))

    _check_inference(
        bb,
        relax.op.vision.all_class_non_max_suppression(boxes, scores, output_format="onnx"),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((batch_size * num_classes * num_boxes, 3), "int64"),
                relax.TensorStructInfo((1,), "int64"),
            ]
        ),
    )

    _check_inference(
        bb,
        relax.op.vision.all_class_non_max_suppression(boxes, scores, output_format="tensorflow"),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((batch_size, num_classes * num_boxes, 2), "int64"),
                relax.TensorStructInfo(
                    (
                        batch_size,
                        num_classes * num_boxes,
                    ),
                    "float32",
                ),
                relax.TensorStructInfo((batch_size,), "int64"),
            ]
        ),
    )


if __name__ == "__main__":
    tvm.testing.main()
