# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""Unit tests for PrimalLayoutTransformToTranspose pass."""
# pylint: disable=not-callable

import tvm
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass import transform as compass_transform


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_primal_layout_transpose_to_transform():
    dtype = "float32"
    ishape = (1, 4, 14, 14)
    wshape = (32, 4, 3, 3)

    def before():
        data = relay.var("data", shape=(ishape), dtype=dtype)
        weight = relay.var("weight", shape=(wshape), dtype=dtype)
        data = relay.layout_transform(data, src_layout="NCHW", dst_layout="NHWC")
        weight = relay.layout_transform(weight, src_layout="OIHW", dst_layout="OWHI")
        out = relay.nn.conv2d(
            data,
            weight,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="OWHI",
        )
        out = relay.Function(relay.analysis.free_vars(out), out)
        return out

    def expected():
        data = relay.var("data", shape=(ishape), dtype=dtype)
        weight = relay.var("weight", shape=(wshape), dtype=dtype)
        # "NCHW" -> "NHWC"
        data = relay.transpose(data, axes=[0, 2, 3, 1])
        # "OIHW" -> "OWHI"
        weight = relay.transpose(weight, axes=[0, 3, 2, 1])
        out = relay.nn.conv2d(
            data,
            weight,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="OWHI",
        )
        out = relay.Function(relay.analysis.free_vars(out), out)
        return out

    a = before()
    a = run_opt_pass(a, [compass_transform.PrimalLayoutTransformToTranspose()])
    b = run_opt_pass(expected(), [relay.transform.InferType()])
    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


if __name__ == "__main__":
    test_primal_layout_transpose_to_transform()
