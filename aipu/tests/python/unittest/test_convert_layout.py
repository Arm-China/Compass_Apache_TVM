# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""Test conv2d transform from NHWC to NCHW8c"""
import tvm
from tvm import relay
from tvm.relay import transform, analysis


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


# Pytest Specific Function
def test_conv2d_bn_convert_layout():
    """Check that layout transforms are propagated through bn."""

    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )

        dtype = "float32"
        beta = relay.var("beta", relay.TensorType((64,), dtype))
        gamma = relay.var("gamma", relay.TensorType((64,), dtype))
        moving_mean = relay.var("moving_mean", relay.TensorType((64,), dtype))
        moving_var = relay.var("moving_var", relay.TensorType((64,), dtype))

        y = relay.nn.batch_norm(y, gamma, beta, moving_mean, moving_var, axis=3)
        y = relay.nn.relu(y[0])
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        w = relay.var("weight", shape=(3, 3, 64, 64))
        x1 = relay.layout_transform(x, "NHWC", "NCHW8c")
        w1 = relay.layout_transform(w, "HWIO", "OIHW8o4i")
        y = relay.nn.contrib_conv2d_nchwc(
            x1,
            w1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW8c",
            kernel_layout="OIHW8o4i",
        )

        dtype = "float32"
        beta = relay.var("beta", relay.TensorType((64,), dtype))
        gamma = relay.var("gamma", relay.TensorType((64,), dtype))
        moving_mean = relay.var("moving_mean", relay.TensorType((64,), dtype))
        moving_var = relay.var("moving_var", relay.TensorType((64,), dtype))

        bn = relay.add(moving_var, relay.const(1e-5, dtype))
        bn = relay.sqrt(bn)
        bn = relay.divide(relay.const(1, dtype), bn)
        bn_temp = relay.multiply(bn, gamma)
        bn = relay.expand_dims(bn_temp, axis=0, num_newaxis=3)
        bn = relay.layout_transform(bn, src_layout="NHWC", dst_layout="NCHW8c")

        y = relay.multiply(y, bn)
        bn = relay.negative(moving_mean)
        bn = relay.multiply(bn, bn_temp)
        bn = relay.add(bn, beta)
        bn = relay.expand_dims(bn, axis=0, num_newaxis=3)
        bn = relay.layout_transform(bn, src_layout="NHWC", dst_layout="NCHW8c")

        y = relay.add(y, bn)
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, src_layout="NCHW8c", dst_layout="NHWC")

        y = relay.Function([x, w, gamma, beta, moving_mean, moving_var], y)
        return y

    a = before()
    a = run_opt_pass(
        a,
        [
            transform.SimplifyInference(),
            transform.ConvertLayout({"nn.conv2d": ["NCHW8c", "OIHW8o4i"]}),
        ],
    )
    b = run_opt_pass(expected(), [transform.InferType()])

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


if __name__ == "__main__":
    test_conv2d_bn_convert_layout()
