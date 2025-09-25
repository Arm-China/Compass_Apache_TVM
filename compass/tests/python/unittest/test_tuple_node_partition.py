# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=no-self-argument
import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.relax.dpl import is_op, wildcard
from tvm.compass.relax import transform as compass_transform, Compass


def op_pattern(op_name, input_num=1):
    out = is_op(f"relax.{op_name}")(*[wildcard() for _ in range(input_num)])
    annotations = {"root": out}
    return (f"compass.{op_name}", out, annotations)


PATTERNS = (op_pattern("abs"), op_pattern("concat"), op_pattern("reshape", 2))


def test_var_input():
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def main(inp0: R.Tensor((1, 4), dtype="float32"), inp1: R.Tensor((1, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
            with R.dataflow():
                gv: R.Tensor((2, 4), dtype="float32") = R.concat((inp0, inp1), axis=0)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def tvm_compass_subfunc0(inp0: R.Tensor((1, 4), dtype="float32"), inp1: R.Tensor((1, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
            R.func_attr({"Codegen": "compass"})
            # from tvm.script import relax as R

            @R.function
            def gv(inp0_1: R.Tensor((1, 4), dtype="float32"), inp1_1: R.Tensor((1, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
                R.func_attr({"Composite": "compass.concat"})
                with R.dataflow():
                    gv_1: R.Tensor((2, 4), dtype="float32") = R.concat((inp0_1, inp1_1), axis=0)
                    R.output(gv_1)
                return gv_1

            gv_1: R.Tensor((2, 4), dtype="float32") = gv(inp0, inp1)
            return gv_1

        @R.function
        def main(inp0: R.Tensor((1, 4), dtype="float32"), inp1: R.Tensor((1, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
            cls = Expected
            with R.dataflow():
                gv: R.Tensor((2, 4), dtype="float32") = cls.tvm_compass_subfunc0(inp0, inp1)
                R.output(gv)
            return gv
    # fmt: on

    # Init CompassConfig
    Compass("")
    ir_mod = Module
    mod = relax.transform.FuseOpsByPattern(PATTERNS)(ir_mod)
    mod = compass_transform.FuseTuple()(mod)
    mod = relax.transform.MergeCompositeFunctions()(mod)
    mod = compass_transform.RenameCompassSubfunc()(mod)

    assert relax.analysis.well_formed(mod)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_expr_input():
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def main(inp0: R.Tensor((1, 4), dtype="float32"), inp1: R.Tensor((1, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
            with R.dataflow():
                lv0: R.Tensor((1, 4), dtype="float32") = R.abs(inp0)
                lv1: R.Tensor((1, 4), dtype="float32") = R.abs(inp1)
                gv: R.Tensor((2, 4), dtype="float32") = R.concat((lv0, lv1), axis=0)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def tvm_compass_subfunc0(inp0: R.Tensor((1, 4), dtype="float32"), inp1: R.Tensor((1, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
            R.func_attr({"Codegen": "compass"})
            # from tvm.script import relax as R

            @R.function
            def gv(inp0_1: R.Tensor((1, 4), dtype="float32")) -> R.Tensor((1, 4), dtype="float32"):
                R.func_attr({"Composite": "compass.abs"})
                with R.dataflow():
                    gv_1: R.Tensor((1, 4), dtype="float32") = R.abs(inp0_1)
                    R.output(gv_1)
                return gv_1

            lv: R.Tensor((1, 4), dtype="float32") = gv(inp0)
            lv1: R.Tensor((1, 4), dtype="float32") = gv(inp1)
            # from tvm.script import relax as R

            @R.function
            def gv1(lv_1: R.Tensor((1, 4), dtype="float32"), lv1_1: R.Tensor((1, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
                R.func_attr({"Composite": "compass.concat"})
                with R.dataflow():
                    gv_1: R.Tensor((2, 4), dtype="float32") = R.concat((lv_1, lv1_1), axis=0)
                    R.output(gv_1)
                return gv_1

            gv_1: R.Tensor((2, 4), dtype="float32") = gv1(lv, lv1)
            return gv_1

        @R.function
        def main(inp0: R.Tensor((1, 4), dtype="float32"), inp1: R.Tensor((1, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
            cls = Expected
            with R.dataflow():
                gv: R.Tensor((2, 4), dtype="float32") = cls.tvm_compass_subfunc0(inp0, inp1)
                R.output(gv)
            return gv
    # fmt: on

    # Init CompassConfig
    Compass("")
    ir_mod = Module
    mod = relax.transform.FuseOpsByPattern(PATTERNS)(ir_mod)
    mod = compass_transform.FuseTuple()(mod)
    mod = relax.transform.MergeCompositeFunctions()(mod)
    mod = compass_transform.RenameCompassSubfunc()(mod)

    assert relax.analysis.well_formed(mod)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    test_var_input()
    test_expr_input()
