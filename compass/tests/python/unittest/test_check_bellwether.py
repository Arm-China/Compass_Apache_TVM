# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=no-self-argument
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.compass.relax import analysis as compass_analysis


r_i32 = R.Prim("int32")


def test_multiple_definition():
    @I.ir_module
    class Module:
        @R.function
        def my_add1(i2: r_i32, i3: r_i32) -> r_i32:
            R.func_attr({"Codegen": "byoc"})
            return R.add(i2, i3)

        @R.function
        def my_add2(i2: r_i32, i3: r_i32) -> r_i32:
            R.func_attr({"Codegen": "byoc"})
            return R.add(i2, i3)

        @R.function
        def main(i0: r_i32, i1: r_i32) -> r_i32:
            cls = Module
            with R.dataflow():
                lv = cls.my_add1(i0, i1)
                lv1 = cls.my_add2(lv, i1)
                R.output(lv1)
            return lv1

    assert compass_analysis.check_bellwether(Module, "byoc") is False


def test_multiple_call():
    @I.ir_module
    class Module:
        @R.function
        def my_add(i2: r_i32, i3: r_i32) -> r_i32:
            R.func_attr({"Codegen": "byoc"})
            return R.add(i2, i3)

        @R.function
        def main(i0: r_i32, i1: r_i32) -> r_i32:
            cls = Module
            with R.dataflow():
                lv = cls.my_add(i0, i1)
                lv1 = cls.my_add(lv, i1)
                R.output(lv1)
            return lv1

    assert compass_analysis.check_bellwether(Module, "byoc") is False


def test_argument_count_mismatch():
    @I.ir_module
    class Module:
        @R.function
        def my_add(i2: r_i32, i3: r_i32) -> r_i32:
            R.func_attr({"Codegen": "byoc"})
            return R.add(i2, i3)

        @R.function
        def main(i0: r_i32) -> r_i32:
            with R.dataflow():
                lv = Module.my_add(i0, i0)
                R.output(lv)
            return lv

    assert compass_analysis.check_bellwether(Module, "byoc") is False


def test_argument_not_same():
    @I.ir_module
    class Module:
        @R.function
        def my_add(i2: r_i32, i3: r_i32) -> r_i32:
            R.func_attr({"Codegen": "byoc"})
            return R.add(i2, i3)

        @R.function
        def main(i0: r_i32, i1: r_i32) -> r_i32:
            return Module.my_add(i0, i0)

    assert compass_analysis.check_bellwether(Module, "byoc") is False


def test_local_sub_function():
    @I.ir_module
    class Module:
        @R.function
        def my_add(i2: r_i32, i3: r_i32) -> r_i32:
            R.func_attr({"Codegen": "byoc"})

            @R.function
            def gv(i0: r_i32, i1: r_i32) -> r_i32:
                return R.add(i0, i1)

            return gv(i2, i3)

        @R.function
        def main(i0: r_i32, i1: r_i32) -> r_i32:
            with R.dataflow():
                lv = Module.my_add(i0, i1)
                R.output(lv)
            return lv

    assert compass_analysis.check_bellwether(Module, "byoc") is True


def test_positive_case():
    @I.ir_module
    class Module:
        @R.function
        def my_add(i2: r_i32, i3: r_i32) -> r_i32:
            R.func_attr({"Codegen": "byoc"})
            return R.add(i2, i3)

        @R.function
        def main(i0: r_i32, i1: r_i32) -> r_i32:
            with R.dataflow():
                lv = Module.my_add(i0, i1)
                R.output(lv)
            return lv

    assert compass_analysis.check_bellwether(Module, "byoc") is True


if __name__ == "__main__":
    test_multiple_definition()
    test_multiple_call()
    test_argument_count_mismatch()
    test_argument_not_same()
    test_local_sub_function()
    test_positive_case()
