# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
from tvm import ir, relay
from tvm.relay.backend.contrib.aipu_compass import transform as compass_transform
from tvm.relay import testing as relay_test


# Pytest Specific Function
def test_simplify_pad():
    dtype = "int8"
    dshape = (1, 224, 224, 3)
    kshape = (7, 7, 3, 64)

    data = relay.var("data", shape=dshape, dtype=dtype)
    weight = relay.var("weight", shape=kshape, dtype=dtype)

    def before():
        expr = relay.nn.pad(data, [[0, 0], [3, 3], [3, 3], [0, 0]])
        expr = relay.nn.conv2d(expr, weight, data_layout="NHWC", kernel_layout="HWIO")
        return relay.Function(relay.analysis.free_vars(expr), expr)

    def expected():
        expr = relay.nn.conv2d(data, weight, padding=(3, 3, 3, 3), data_layout="NHWC", kernel_layout="HWIO")
        return relay.Function(relay.analysis.free_vars(expr), expr)

    after = relay_test.run_opt_pass(before(), compass_transform.SimplifyPad())
    expected = relay_test.run_infer_type(expected())
    assert ir.structural_equal(after, expected), f"\nExpected:\n{expected}\n" f"\nAfter:\n{after}\n"


if __name__ == "__main__":
    test_simplify_pad()
