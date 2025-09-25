# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=no-self-argument
import tvm
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.compass.relax import transform as compass_transform


def test_one_in_one_out():
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def tvm_compass_subfunc0(x_1: R.Tensor((1, 3, 224, 224), dtype="float32")) -> R.Tensor((1, 3, 224, 224), dtype="float32"):
            R.func_attr({"Codegen": "compass", "quant_infos": []})
            lv: R.Tensor((1, 3, 224, 224), dtype="uint8") = R.quantize(x_1, R.const(0.037463810294866562, "float32"), R.const(57, "int32"), out_dtype="uint8", axis=1)
            lv_1: R.Tensor((1, 3, 224, 224), dtype="uint8") = R.add(lv, lv)
            gv_1: R.Tensor((1, 3, 224, 224), dtype="float32") = R.dequantize(lv_1, R.const(0.30345085263252258, "float32"), R.const(37, "int32"), out_dtype="float32", axis=1)
            return gv_1

        @R.function
        def main(x_1: R.Tensor((1, 3, 224, 224), dtype="float32")) -> R.Tensor((1, 3, 224, 224), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = Module
            with R.dataflow():
                gv: R.Tensor((1, 3, 224, 224), dtype="float32") = cls.tvm_compass_subfunc0(x_1)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def tvm_compass_subfunc0(x_1: R.Tensor((1, 3, 224, 224), dtype="uint8")) -> R.Tensor((1, 3, 224, 224), dtype="uint8"):
            R.func_attr({"Codegen": "compass", "quant_infos": []})
            lv_1: R.Tensor((1, 3, 224, 224), dtype="uint8") = R.add(x_1, x_1)
            return lv_1

        @R.function
        def main(x_1: R.Tensor((1, 3, 224, 224), dtype="float32")) -> R.Tensor((1, 3, 224, 224), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                lv: R.Tensor((1, 3, 224, 224), dtype="uint8") = R.quantize(x_1, R.const(0.037463810294866562, "float32"), R.const(57, "int32"), out_dtype="uint8", axis=1)
                lv1: R.Tensor((1, 3, 224, 224), dtype="uint8") = cls.tvm_compass_subfunc0(lv)
                gv: R.Tensor((1, 3, 224, 224), dtype="float32") = R.dequantize(lv1, R.const(0.30345085263252258, "float32"), R.const(37, "int32"), out_dtype="float32", axis=1)
                R.output(gv)
            return gv
    # fmt: on

    mod = compass_transform.ExtractInOutQuantOps()(Module)
    assert tvm.relax.analysis.well_formed(mod)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_one_in_two_out():
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def tvm_compass_subfunc0(x_1: R.Tensor((1, 3, 224, 224), dtype="float32")) -> R.Tuple(R.Tensor((1, 3, 224, 224), dtype="float32"), R.Tensor((1, 3, 224, 224), dtype="float32")):
            R.func_attr({"Codegen": "compass", "quant_infos": []})
            lv: R.Tensor((1, 3, 224, 224), dtype="uint8") = R.quantize(x_1, R.const(0.037463810294866562, "float32"), R.const(57, "int32"), out_dtype="uint8", axis=1)
            lv_1: R.Tensor((1, 3, 224, 224), dtype="uint8") = R.add(lv, lv)
            lv_2: R.Tensor((1, 3, 224, 224), dtype="uint8") = R.add(lv_1, lv)
            gv_1: R.Tensor((1, 3, 224, 224), dtype="float32") = R.dequantize(lv_1, R.const(0.30345085263252258, "float32"), R.const(37, "int32"), out_dtype="float32", axis=1)
            gv_2: R.Tensor((1, 3, 224, 224), dtype="float32") = R.dequantize(lv_2, R.const(0.30345085263252258, "float32"), R.const(37, "int32"), out_dtype="float32", axis=1)
            gv: R.Tuple(R.Tensor((1, 3, 224, 224), dtype="float32"), R.Tensor((1, 3, 224, 224), dtype="float32")) = gv_1, gv_2
            return gv

        @R.function
        def main(x_1: R.Tensor((1, 3, 224, 224), dtype="float32")) -> R.Tuple(R.Tensor((1, 3, 224, 224), dtype="float32"), R.Tensor((1, 3, 224, 224), dtype="float32")):
            R.func_attr({"num_input": 1})
            cls = Module
            with R.dataflow():
                gv: R.Tuple(R.Tensor((1, 3, 224, 224), dtype="float32"), R.Tensor((1, 3, 224, 224), dtype="float32")) = cls.tvm_compass_subfunc0(x_1)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def tvm_compass_subfunc0(x_1: R.Tensor((1, 3, 224, 224), dtype="uint8")) -> R.Tuple(R.Tensor((1, 3, 224, 224), dtype="uint8"), R.Tensor((1, 3, 224, 224), dtype="uint8")):
            R.func_attr({"Codegen": "compass", "quant_infos": []})
            lv_1: R.Tensor((1, 3, 224, 224), dtype="uint8") = R.add(x_1, x_1)
            lv_2: R.Tensor((1, 3, 224, 224), dtype="uint8") = R.add(lv_1, x_1)
            gv: R.Tuple(R.Tensor((1, 3, 224, 224), dtype="uint8"), R.Tensor((1, 3, 224, 224), dtype="uint8")) = lv_1, lv_2
            return gv

        @R.function
        def main(x_1: R.Tensor((1, 3, 224, 224), dtype="float32")) -> R.Tuple(R.Tensor((1, 3, 224, 224), dtype="float32"), R.Tensor((1, 3, 224, 224), dtype="float32")):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                lv: R.Tensor((1, 3, 224, 224), dtype="uint8") = R.quantize(x_1, R.const(0.037463810294866562, "float32"), R.const(57, "int32"), out_dtype="uint8", axis=1)
                lv1: R.Tuple(R.Tensor((1, 3, 224, 224), dtype="uint8"), R.Tensor((1, 3, 224, 224), dtype="uint8")) = cls.tvm_compass_subfunc0(lv)
                lv2: R.Tensor((1, 3, 224, 224), dtype="uint8") = lv1[0]
                lv3: R.Tensor((1, 3, 224, 224), dtype="float32") = R.dequantize(lv2, R.const(0.30345085263252258, "float32"), R.const(37, "int32"), out_dtype="float32", axis=1)
                lv4: R.Tensor((1, 3, 224, 224), dtype="uint8") = lv1[1]
                lv5: R.Tensor((1, 3, 224, 224), dtype="float32") = R.dequantize(lv4, R.const(0.30345085263252258, "float32"), R.const(37, "int32"), out_dtype="float32", axis=1)
                gv: R.Tuple(R.Tensor((1, 3, 224, 224), dtype="float32"), R.Tensor((1, 3, 224, 224), dtype="float32")) = lv3, lv5
                R.output(gv)
            return gv
    # fmt: on

    mod = compass_transform.ExtractInOutQuantOps()(Module)
    assert tvm.relax.analysis.well_formed(Module)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    test_one_in_one_out()
    test_one_in_two_out()
