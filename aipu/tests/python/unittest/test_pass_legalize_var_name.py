# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
import tvm
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass import transform as compass_transform


# Pytest Specific Function
def test_legalize_var_name():
    input0 = relay.var("input-0", shape=[1, 2, 3, 4])
    input1 = relay.var("input-1", shape=[4])
    out = input0 + input1
    mod = tvm.IRModule.from_expr(out)
    mod = compass_transform.LegalizeVarName()(mod)

    new_vars = mod["main"].params
    assert new_vars[0].name_hint == "input_0"
    assert new_vars[1].name_hint == "input_1"


if __name__ == "__main__":
    test_legalize_var_name()
