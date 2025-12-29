# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
# pylint: disable=no-self-argument
import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.compass.relax import transform as compass_transform


t_i32 = R.Tensor((2, 3), "int32")


# Pytest Specific Function
def test_find_postprocess_function():
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def main(i0: t_i32, i1: t_i32, i2: t_i32) -> t_i32:
            with R.dataflow():
                add0 = R.add(i0, relax.const(1, "int32"))
                add01 = R.add(add0, relax.const(1, "int32"))
                lv10 = R.nn.relu(i1)
                lv11 = R.negative(lv10)
                lv12 = R.abs(lv11)
                lv20 = R.nn.relu(i2)
                lv21 = R.abs(lv20)
                add1 = R.add(lv11, lv21)
                add2 = R.add(lv12, add1)
                gv = R.add(add01, add2)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def post_process_func(i0: R.Tensor((2, 3), dtype="int32"), lv11: R.Tensor((2, 3), dtype="int32"), lv21: R.Tensor((2, 3), dtype="int32"), lv12: R.Tensor((2, 3), dtype="int32")) -> R.Tensor((2, 3), dtype="int32"):
            R.func_attr({"Primitive": 1, "kComposite": "post_func"})
            with R.dataflow():
                add0: R.Tensor((2, 3), dtype="int32") = R.add(i0, R.const(1, "int32"))
                add01: R.Tensor((2, 3), dtype="int32") = R.add(add0, R.const(1, "int32"))
                add1: R.Tensor((2, 3), dtype="int32") = R.add(lv11, lv21)
                add2: R.Tensor((2, 3), dtype="int32") = R.add(lv12, add1)
                gv: R.Tensor((2, 3), dtype="int32") = R.add(add01, add2)
                R.output(gv)
            return gv

        @R.function
        def main(i0: R.Tensor((2, 3), dtype="int32"), i1: R.Tensor((2, 3), dtype="int32"), i2: R.Tensor((2, 3), dtype="int32")) -> R.Tensor((2, 3), dtype="int32"):
            with R.dataflow():
                lv10: R.Tensor((2, 3), dtype="int32") = R.nn.relu(i1)
                lv11: R.Tensor((2, 3), dtype="int32") = R.negative(lv10)
                lv12: R.Tensor((2, 3), dtype="int32") = R.abs(lv11)
                lv20: R.Tensor((2, 3), dtype="int32") = R.nn.relu(i2)
                lv21: R.Tensor((2, 3), dtype="int32") = R.abs(lv20)
                out: R.Tensor((2, 3), dtype="int32") = Expected.post_process_func(i0, lv11, lv21, lv12)
                R.output(out)
            return out
    # fmt: on

    def _check(var2val, node):
        if not isinstance(node, relax.Call):
            return False
        return node.op.name == "relax.abs"

    mod = compass_transform.GetPostProcessFunction(_check)(Module)
    assert relax.analysis.well_formed(mod)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    test_find_postprocess_function()
