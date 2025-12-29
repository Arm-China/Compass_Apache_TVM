# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
# pylint: disable=no-self-argument
import tvm
from tvm.relax.dpl import rewrite_call
from tvm.script import relax as R, ir as I, tir as T
from tvm.compass.relax.transform.pattern_rewrites import ExtractNegativePadFromConv2d


def test_extract_negative_pad_from_conv2d():
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32"), w1: R.Tensor((6, 3, 3, 3), dtype="float32")) -> R.Tensor((1, 6, 6, 6), dtype="float32"):
            with R.dataflow():
                gv = R.nn.conv2d(input_1, w1, strides=[1, 1], padding=[-1, -1, -1, -1])
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32"), w1: R.Tensor((6, 3, 3, 3), dtype="float32")) -> R.Tensor((1, 6, 6, 6), dtype="float32"):
            with R.dataflow():
                lv = R.nn.pad(input_1, pad_width=[T.int64(0), T.int64(0), T.int64(0), T.int64(0), T.int64(-1), T.int64(-1), T.int64(-1), T.int64(-1)], pad_value=0.0, pad_mode="constant")
                gv = R.nn.conv2d(lv, w1, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
                R.output(gv)
            return gv
    # fmt: on

    mod = Module
    mod["main"] = rewrite_call(*ExtractNegativePadFromConv2d().pr, mod["main"])
    assert tvm.relax.analysis.well_formed(mod)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    test_extract_negative_pad_from_conv2d()
