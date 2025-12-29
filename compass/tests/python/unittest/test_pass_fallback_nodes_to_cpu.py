# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
# pylint: disable=no-self-argument
import tvm
from tvm.script import relax as R, ir as I
from tvm.compass.relax import Compass


def test_pass_fallback_nodes_to_cpu():
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((1000, 1024), dtype="float32"), y: R.Tensor((1024, 1024), dtype="float32")) -> R.Tensor((1000, 1024), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1000, 1024), dtype="float32") = R.matmul(x, y)
                lv_1: R.Tensor((1000, 1024), dtype="float32") = R.nn.softmax(lv)
                lv_2: R.Tensor((1000, 1024), dtype="float32") = R.add(lv_1, x)
                gv: R.Tensor((1000, 1024), dtype="float32") = R.nn.relu(lv_2)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected1:
        @R.function
        def tvm_compass_subfunc0(gv: R.Tensor((1000, 1024), dtype="float32"), x: R.Tensor((1000, 1024), dtype="float32")) -> R.Tensor((1000, 1024), dtype="float32"):
            R.func_attr({"Codegen": "compass"})
            lv_1: R.Tensor((1000, 1024), dtype="float32") = R.nn.softmax(gv, axis=1)
            # from tvm.script import relax as R

            @R.function
            def gv1(lv_1_1: R.Tensor((1000, 1024), dtype="float32"), x_1: R.Tensor((1000, 1024), dtype="float32")) -> R.Tensor((1000, 1024), dtype="float32"):
                R.func_attr({"Composite": "compass.eltwise_relu"})
                with R.dataflow():
                    lv_2: R.Tensor((1000, 1024), dtype="float32") = R.add(lv_1_1, x_1)
                    gv_1: R.Tensor((1000, 1024), dtype="float32") = R.nn.relu(lv_2)
                    R.output(gv_1)
                return gv_1

            gv_1: R.Tensor((1000, 1024), dtype="float32") = gv1(lv_1, x)
            return gv_1

        @R.function
        def main(x: R.Tensor((1000, 1024), dtype="float32"), y: R.Tensor((1024, 1024), dtype="float32")) -> R.Tensor((1000, 1024), dtype="float32"):
            with R.dataflow():
                gv: R.Tensor((1000, 1024), dtype="float32") = R.matmul(x, y, out_dtype="void")
                gv_1: R.Tensor((1000, 1024), dtype="float32") = Expected1.tvm_compass_subfunc0(gv, x)
                R.output(gv_1)
            return gv_1

    @I.ir_module
    class Expected2:
        @R.function
        def tvm_compass_subfunc0(x: R.Tensor((1000, 1024), dtype="float32"), y: R.Tensor((1024, 1024), dtype="float32")) -> R.Tensor((1000, 1024), dtype="float32"):
            R.func_attr({"Codegen": "compass"})
            lv: R.Tensor((1000, 1024), dtype="float32") = R.matmul(x, y, out_dtype="void")
            gv: R.Tensor((1000, 1024), dtype="float32") = R.nn.softmax(lv, axis=1)
            return gv

        @R.function
        def main(x: R.Tensor((1000, 1024), dtype="float32"), y: R.Tensor((1024, 1024), dtype="float32")) -> R.Tensor((1000, 1024), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1000, 1024), dtype="float32") = Expected2.tvm_compass_subfunc0(x, y)
                lv_2: R.Tensor((1000, 1024), dtype="float32") = R.add(lv, x)
                gv: R.Tensor((1000, 1024), dtype="float32") = R.nn.relu(lv_2)
                R.output(gv)
            return gv
    # fmt: on

    # Init CompassConfig
    cfg_str = """
    [Common]
    dump_annotation_graph = True
    """
    compass = Compass(cfg_str)
    compass.ir_mod = Module
    compass.partition(fallback_nodes={"main": ["lv"]})
    tvm.ir.assert_structural_equal(compass.ir_mod, Expected1)
    compass.ir_mod = Module
    compass.partition(fallback_nodes={"main": ["gv"]})
    tvm.ir.assert_structural_equal(compass.ir_mod, Expected2)


if __name__ == "__main__":
    test_pass_fallback_nodes_to_cpu()
