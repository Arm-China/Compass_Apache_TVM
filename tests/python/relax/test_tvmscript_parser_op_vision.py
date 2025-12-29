# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
from typing import Optional, Union

import tvm
import tvm.script
import tvm.testing
from tvm import IRModule, relax
from tvm.script import relax as R


def _check(
    parsed: Union[relax.Function, IRModule],
    expect: Optional[Union[relax.Function, IRModule]],
):
    test = parsed.script(show_meta=True)
    roundtrip_mod = tvm.script.from_source(test)
    tvm.ir.assert_structural_equal(parsed, roundtrip_mod)
    if expect:
        tvm.ir.assert_structural_equal(parsed, expect)


def test_all_class_non_max_suppression():
    @R.function
    def foo(
        boxes: R.Tensor((10, 5, 4), "int64"),
        scores: R.Tensor((10, 8, 5), "float32"),
    ) -> R.Tuple(R.Tensor((400, 3), "int64"), R.Tensor((1,), "int64")):
        gv: R.Tuple(
            R.Tensor((400, 3), "int64"), R.Tensor((1,), "int64")
        ) = R.vision.all_class_non_max_suppression(
            boxes,
            scores,
        )
        return gv

    boxes = relax.Var("boxes", R.Tensor((10, 5, 4), "int64"))
    scores = relax.Var("scores", R.Tensor((10, 8, 5), "float32"))

    bb = relax.BlockBuilder()
    with bb.function("foo", [boxes, scores]):
        gv = bb.emit(relax.op.vision.all_class_non_max_suppression(boxes, scores))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


if __name__ == "__main__":
    tvm.testing.main()
