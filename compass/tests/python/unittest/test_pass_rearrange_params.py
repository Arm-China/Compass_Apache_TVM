# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
# pylint: disable=no-self-argument
import tvm
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.compass.relax import transform as compass_transform


def test_rearrange_params():

    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def tvm_compass_subfunc0(z: R.Tensor((1024,), dtype="int32"), x: R.Tensor((1,), dtype="int32"), y: R.Tensor((10, 1024), dtype="int32")) -> R.Tensor((10, 1024), dtype="int32"):
            R.func_attr({"Codegen": "compass"})
            # from tvm.script import relax as R

            @R.function
            def gv(z_1: R.Tensor((1024,), dtype="int32")) -> R.Tensor((1024,), dtype="int32"):
                R.func_attr({"Composite": "compass.add"})
                with R.dataflow():
                    gv_1: R.Tensor((1024,), dtype="int32") = R.add(z_1, R.const(1, "int32"))
                    R.output(gv_1)
                return gv_1

            lv_z: R.Tensor((1024,), dtype="int32") = gv(z)
            # from tvm.script import relax as R

            @R.function
            def gv1(y: R.Tensor((10, 1024), dtype="int32"), x_1: R.Tensor((1,), dtype="int32")) -> R.Tensor((10, 1024), dtype="int32"):
                R.func_attr({"Composite": "compass.add"})
                with R.dataflow():
                    x_2: R.Tensor((10, 1024), dtype="int32") = R.tile(x_1, repeats=[10, 1024])
                    gv_1: R.Tensor((10, 1024), dtype="int32") = R.add(x_2, y)
                    R.output(gv_1)
                return gv_1

            lv_xy: R.Tensor((10, 1024), dtype="int32") = gv1(y, x)
            # from tvm.script import relax as R

            @R.function
            def gv2(lv_1: R.Tensor((10, 1024), dtype="int32"), z: R.Tensor((1024,), dtype="int32")) -> R.Tensor((10, 1024), dtype="int32"):
                R.func_attr({"Composite": "compass.add"})
                with R.dataflow():
                    z_2: R.Tensor((10240,), dtype="int32") = R.tile(z, repeats=[10, ])
                    z_3: R.Tensor((10, 1024), dtype="int32") = R.reshape(z_2, shape=[10, 1024])
                    gv_1: R.Tensor((10, 1024), dtype="int32") = R.add(z_3, lv_1)
                    R.output(gv_1)
                return gv_1

            gv_1: R.Tensor((10, 1024), dtype="int32") = gv2(lv_xy, lv_z)
            return gv_1

        @R.function
        def main(x: R.Tensor((1,), dtype="int32"), y: R.Tensor((10, 1024), dtype="int32"), z: R.Tensor((1024,), dtype="int32")) -> R.Tensor((10, 1024), dtype="int32"):
            cls = Module
            with R.dataflow():
                gv: R.Tensor((10, 1024), dtype="int32") = cls.tvm_compass_subfunc0(z, x, y)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def tvm_compass_subfunc0(x: R.Tensor((1,), dtype="int32"), y: R.Tensor((10, 1024), dtype="int32"), z: R.Tensor((1024,), dtype="int32")) -> R.Tensor((10, 1024), dtype="int32"):
            R.func_attr({"Codegen": "compass"})
            # from tvm.script import relax as R

            @R.function
            def gv(z_1: R.Tensor((1024,), dtype="int32")) -> R.Tensor((1024,), dtype="int32"):
                R.func_attr({"Composite": "compass.add"})
                with R.dataflow():
                    gv_1: R.Tensor((1024,), dtype="int32") = R.add(z_1, R.const(1, "int32"))
                    R.output(gv_1)
                return gv_1

            lv_z: R.Tensor((1024,), dtype="int32") = gv(z)
            # from tvm.script import relax as R

            @R.function
            def gv1(y_1: R.Tensor((10, 1024), dtype="int32"), x_1: R.Tensor((1,), dtype="int32")) -> R.Tensor((10, 1024), dtype="int32"):
                R.func_attr({"Composite": "compass.add"})
                with R.dataflow():
                    x_2: R.Tensor((10, 1024), dtype="int32") = R.tile(x_1, repeats=[10, 1024])
                    gv_1: R.Tensor((10, 1024), dtype="int32") = R.add(x_2, y_1)
                    R.output(gv_1)
                return gv_1

            lv_xy: R.Tensor((10, 1024), dtype="int32") = gv1(y, x)
            # from tvm.script import relax as R

            @R.function
            def gv2(lv_1: R.Tensor((10, 1024), dtype="int32"), z_1: R.Tensor((1024,), dtype="int32")) -> R.Tensor((10, 1024), dtype="int32"):
                R.func_attr({"Composite": "compass.add"})
                with R.dataflow():
                    z_2: R.Tensor((10240,), dtype="int32") = R.tile(z_1, repeats=[10])
                    z_3: R.Tensor((10, 1024), dtype="int32") = R.reshape(z_2, R.shape([10, 1024]))
                    gv_1: R.Tensor((10, 1024), dtype="int32") = R.add(z_3, lv_1)
                    R.output(gv_1)
                return gv_1

            gv_1: R.Tensor((10, 1024), dtype="int32") = gv2(lv_xy, lv_z)
            return gv_1

        @R.function
        def main(x: R.Tensor((1,), dtype="int32"), y: R.Tensor((10, 1024), dtype="int32"), z: R.Tensor((1024,), dtype="int32")) -> R.Tensor((10, 1024), dtype="int32"):
            cls = Expected
            with R.dataflow():
                gv: R.Tensor((10, 1024), dtype="int32") = cls.tvm_compass_subfunc0(x, y, z)
                R.output(gv)
            return gv
    # fmt: on

    mod = compass_transform.RearrangeParams()(Module)
    assert tvm.relax.analysis.well_formed(mod)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    test_rearrange_params()
