# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import relax
from tvm.compass.relax import testing


@pytest.mark.parametrize("engine", ("opt_float", "opt_int", "gt"))
def test_engine(engine):
    compare_threshold = 0.706 if engine != "opt_float" else 1.0
    data_shape = [1, 128, 128, 32]
    data = relax.Var("scale", relax.TensorStructInfo(data_shape, "float32"))

    bb = relax.BlockBuilder()
    with bb.function("main", [data]):
        with bb.dataflow() as _:
            lv0 = relax.op.sign(data)
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)
    mod = bb.get()

    data_input = np.random.random(data_shape).astype("float32")
    testing.compare_relax_result(mod, data_input, engine=engine, compare_threshold=compare_threshold)


if __name__ == "__main__":
    test_engine("gt")
    test_engine("opt_float")
    test_engine("opt_int")
