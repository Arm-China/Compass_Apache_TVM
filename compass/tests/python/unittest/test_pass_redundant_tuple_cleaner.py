# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
# pylint: disable=no-self-argument
import tvm
from tvm import relax
from tvm.script import relax as R, ir as I
from tvm.compass.relax.transform.build_compass_subgraph import RedundantTupleCleaner


def test_redundant_tuple_cleaner():
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((1, 2), dtype="float32"), y: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((6, 2), dtype="float32"):
            with R.dataflow():
                t0 = R.add(x, y)
                t1 = R.subtract(x, y)
                t2 = R.multiply(x, y)
                tup0 = t0, t1, t2
                l0 = tup0[0]
                l1 = tup0[1]
                l2 = tup0[2]
                tup1 = l0, l1, l2
                tup2 = l2, l0, l1
                lv2 = R.concat(tup0, axis=0)
                lv3 = R.concat(tup1, axis=0)
                lv4 = R.add(lv2, lv3)
                lv5 = R.concat(tup2, axis=0)
                gv = R.add(lv4, lv5)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2), dtype="float32"), y: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((6, 2), dtype="float32"):
            with R.dataflow():
                t0 = R.add(x, y)
                t1 = R.subtract(x, y)
                t2 = R.multiply(x, y)
                tup0 = t0, t1, t2
                tup2 = t2, t0, t1
                lv2 = R.concat(tup0, axis=0)
                lv3 = R.concat(tup0, axis=0)
                lv4 = R.add(lv2, lv3)
                lv5 = R.concat(tup2, axis=0)
                gv = R.add(lv4, lv5)
                R.output(gv)
            return gv
    # fmt: on

    mod = Module
    var2val = relax.analysis.get_var2val(mod["main"])
    mod["main"] = RedundantTupleCleaner(var2val).visit_expr(mod["main"])

    mod = relax.transform.RemoveUnusedOutputs()(mod)
    assert tvm.relax.analysis.well_formed(mod)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    test_redundant_tuple_cleaner()
