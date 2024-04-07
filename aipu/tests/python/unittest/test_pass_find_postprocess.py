# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import tvm
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass import transform as compass_transform


# Pytest Specific Function
def test_find_postprocess_function():
    inp0 = relay.var("input0", shape=[1, 1, 1, 1])
    inp1 = relay.var("input1", shape=[1])
    conv0_w = relay.var("conv0_weight", shape=[1, 1, 1, 1])
    conv1_w = relay.var("conv1_weight", shape=[1, 1, 1, 1])
    conv2_w = relay.var("conv2_weight", shape=[1, 1, 1, 1])
    conv3_w = relay.var("conv3_weight", shape=[1, 1, 1, 1])
    conv0 = relay.nn.conv2d(inp0, conv0_w)
    conv1 = relay.nn.conv2d(conv0, conv1_w)
    conv2 = relay.nn.conv2d(conv0, conv2_w)
    conv3 = relay.nn.conv2d(inp0, conv3_w)
    out = conv1 + conv2 + conv3 + inp1
    mod = tvm.IRModule.from_expr(out)

    def expected():
        var0 = relay.var("var0", shape=[1, 1, 1, 1])
        var1 = relay.var("var1", shape=[1, 1, 1, 1])
        var2 = relay.var("var2", shape=[1, 1, 1, 1])
        var3 = relay.var("var3", shape=[1])
        out = var0 + var1 + var2 + var3
        return out

    def _check(node):
        if not isinstance(node, relay.Call):
            return False
        return node.op == relay.op.get("nn.conv2d")

    mod_pass = compass_transform.GetPostProcessFunction(_check)
    update_mod = mod_pass(mod)
    gvar = update_mod["main"].body.op
    assert tvm.ir.structural_equal(update_mod[gvar].body, expected(), True)


if __name__ == "__main__":
    test_find_postprocess_function()
