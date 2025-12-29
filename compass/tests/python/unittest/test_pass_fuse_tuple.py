# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm import relax
from tvm.compass.relax import testing


def test_tuple_fusion():
    data_shape = (4, 4)
    x = relax.Var("x", relax.TensorStructInfo(data_shape, "float32"))

    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow() as _:
            add = relax.op.add(x, x)
            tup = relax.Tuple([add, x])
            concat = relax.op.concat(tup, axis=1)
            gv0 = bb.emit_output(concat)
        bb.emit_func_output(gv0)
    mod = bb.get()

    data_input = np.random.random(data_shape).astype("float32")
    testing.compare_relax_opt_float_result(mod, data_input)


def test_tuple_get_item_fusion():
    data_shape = (4, 4)
    x = relax.Var("x", relax.TensorStructInfo(data_shape, "float32"))
    y = relax.Var("y", relax.TensorStructInfo(data_shape, "float32"))

    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        with bb.dataflow() as _:
            add = relax.op.add(x, x)
            tup0 = relax.Tuple([add, x])
            tup0_0 = relax.TupleGetItem(tup0, 0)
            tup1 = relax.Tuple([tup0_0, y])
            gv0 = bb.emit_output(tup1)
        bb.emit_func_output(gv0)
    mod = bb.get()

    data_input = [np.random.random(data_shape).astype("float32") for _ in range(2)]
    testing.compare_relax_opt_float_result(mod, *data_input)


if __name__ == "__main__":
    test_tuple_fusion()
    test_tuple_get_item_fusion()
