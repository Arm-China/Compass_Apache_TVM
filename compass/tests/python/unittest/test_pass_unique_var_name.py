# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
# pylint: disable=no-self-argument
import tvm
from tvm.script import relax as R, ir as I
from tvm.compass.relax import transform as compass_transform


def test_pass_unique_var_name():
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((1000, 1024), dtype="float32"), y: R.Tensor((1024, 1024), dtype="float32")) -> R.Tensor((1000, 1024), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1000, 1024), dtype="float32") = R.matmul(x, y)
                lv: R.Tensor((1000, 1024), dtype="float32") = R.nn.softmax(lv)
                lv: R.Tensor((1000, 1024), dtype="float32") = R.add(lv, x)
                gv: R.Tensor((1000, 1024), dtype="float32") = R.nn.relu(lv)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1000, 1024), dtype="float32"), y: R.Tensor((1024, 1024), dtype="float32")) -> R.Tensor((1000, 1024), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1000, 1024), dtype="float32") = R.matmul(x, y, out_dtype="void")
                lv_1: R.Tensor((1000, 1024), dtype="float32") = R.nn.softmax(lv, axis=-1)
                lv_2: R.Tensor((1000, 1024), dtype="float32") = R.add(lv_1, x)
                gv: R.Tensor((1000, 1024), dtype="float32") = R.nn.relu(lv_2)
                R.output(gv)
            return gv

    # fmt: on

    mod = Module
    update_mod = compass_transform.UniqueVarName()(mod)
    tvm.ir.assert_structural_equal(update_mod, Expected)
    assert update_mod["main"].body.blocks[0].bindings[1].var.name_hint == "lv_1"
    assert update_mod["main"].body.blocks[0].bindings[2].var.name_hint == "lv_2"


if __name__ == "__main__":
    test_pass_unique_var_name()
