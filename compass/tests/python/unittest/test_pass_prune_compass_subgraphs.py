# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=no-self-argument
import tvm
from tvm import relax
from tvm.script import relax as R, ir as I
from tvm.relax.dpl import is_op, wildcard
from tvm.compass.relax import transform as compass_transform, Compass


def pattern1():
    out = is_op("relax.matmul")(wildcard(), wildcard())
    annotations = {"root": out}
    return ("compass.matmul", out, annotations)


def pattern2():
    out = is_op("relax.reshape")(wildcard(), wildcard())
    annotations = {"root": out}
    return ("compass.reshape", out, annotations)


def test_pass_prune_compass_subgraphs():
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((1000, 1024), dtype="int32"), y: R.Tensor((1024, 1024), dtype="int32")) -> R.Tensor((1000, 512, 2), dtype="int32"):
            with R.dataflow():
                lv: R.Tensor((1000, 1024), dtype="int32") = R.matmul(x, y, out_dtype="void")
                lv1: R.Tensor((1000, 1024), dtype="int32") = R.nn.relu(lv)
                gv: R.Tensor((1000, 512, 2), dtype="int32") = R.reshape(lv1, R.shape([1000, 512, 2]))
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def fused_relax_matmul1_compass(x: R.Tensor((1000, 1024), dtype="int32"), y: R.Tensor((1024, 1024), dtype="int32")) -> R.Tensor((1000, 1024), dtype="int32"):
            R.func_attr({"Codegen": "compass"})
            # from tvm.script import relax as R

            @R.function
            def gv(x_1: R.Tensor((1000, 1024), dtype="int32"), y_1: R.Tensor((1024, 1024), dtype="int32")) -> R.Tensor((1000, 1024), dtype="int32"):
                R.func_attr({"Composite": "compass.matmul"})
                with R.dataflow():
                    gv_1: R.Tensor((1000, 1024), dtype="int32") = R.matmul(x_1, y_1, out_dtype="void")
                    R.output(gv_1)
                return gv_1

            gv_1: R.Tensor((1000, 1024), dtype="int32") = gv(x, y)
            return gv_1

        @R.function
        def main(x: R.Tensor((1000, 1024), dtype="int32"), y: R.Tensor((1024, 1024), dtype="int32")) -> R.Tensor((1000, 512, 2), dtype="int32"):
            with R.dataflow():
                lv: R.Tensor((1000, 1024), dtype="int32") = Expected.fused_relax_matmul1_compass(x, y)
                lv1: R.Tensor((1000, 1024), dtype="int32") = R.nn.relu(lv)
                gv: R.Tensor((1000, 512, 2), dtype="int32") = R.reshape(lv1, R.shape([1000, 512, 2]))
                R.output(gv)
            return gv
    # fmt: on

    # Init CompassConfig
    Compass("")
    mod = Module
    mod = relax.transform.FuseOpsByPattern((pattern1(), pattern2()))(mod)
    mod = relax.transform.MergeCompositeFunctions()(mod)
    mod = compass_transform.PruneCompassSubGraphs()(mod)
    assert relax.analysis.well_formed(mod)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    test_pass_prune_compass_subgraphs()
