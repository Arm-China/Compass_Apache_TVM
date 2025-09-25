# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
from tvm import relax
from tvm.compass.relax import transform as compass_transform


def test_legalize_var_name():
    input0_shape = [1, 128, 128, 32]
    input0 = relax.Var("input-0", relax.TensorStructInfo(input0_shape, "float32"))
    input1_shape = [1]
    input1 = relax.Var("input.1", relax.TensorStructInfo(input1_shape, "float32"))
    input2_shape = [32]
    input2 = relax.Var("input:2", relax.TensorStructInfo(input2_shape, "float32"))

    bb = relax.BlockBuilder()
    with bb.function("main", [input0, input1, input2]):
        with bb.dataflow() as _:
            lv0 = input0 + input1 + input2
            gv0 = bb.emit_output(lv0)
        bb.emit_func_output(gv0)
    ir_mod = bb.get()

    mod = compass_transform.LegalizeVarName()(ir_mod)
    new_vars = mod["main"].params
    assert new_vars[0].name_hint == "input_0"
    assert new_vars[1].name_hint == "input_1"
    assert new_vars[2].name_hint == "input_2"


if __name__ == "__main__":
    test_legalize_var_name()
