# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
import tvm
import numpy as np
from tvm import relay


def dequantize(data, scale, zp):
    return scale * (np.asarray(data) - zp)


def generate_golden_output(x_data, dequantized_x, alpha, o_scale, o_zero_point, i_zero_point):
    q_min = np.iinfo(np.uint8).min
    q_max = np.iinfo(np.uint8).max

    prod = np.multiply(dequantized_x, alpha)
    prod = np.clip(np.around(prod / o_scale + o_zero_point), q_min, q_max)
    requantized = np.clip(np.round(dequantized_x / o_scale + o_zero_point), q_min, q_max)

    output = np.where(x_data < i_zero_point, prod, requantized)
    return output


def test_qnn_prelu():
    data_dtype = "uint8"
    input_scale = 0.125
    input_zero_point = 60
    alpha_scale = 0.234
    alpha_zero_point = 86
    output_scale = 1.98
    output_zero_point = 17
    axis = 0

    x = relay.var("x", shape=(4,), dtype=data_dtype)
    alpha = relay.var("alpha", shape=(4,), dtype=data_dtype)
    y = relay.qnn.op.prelu(
        x=x,
        alpha=alpha,
        input_scale=relay.const(input_scale, "float32"),
        input_zero_point=relay.const(input_zero_point, "int32"),
        alpha_scale=relay.const(alpha_scale, "float32"),
        alpha_zero_point=relay.const(alpha_zero_point, "int32"),
        output_scale=relay.const(output_scale, "float32"),
        output_zero_point=relay.const(output_zero_point, "int32"),
        axis=axis,
    )

    func = relay.Function([x, alpha], y)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_data = np.array((255, 133, 0, 9))
    alpha_data = np.array((1, 2, 3, 4))
    x_dequantized = dequantize(x_data, input_scale, input_zero_point)
    alpha_dequantized = dequantize(alpha_data, alpha_scale, alpha_zero_point)
    golden_output = generate_golden_output(
        x_data, x_dequantized, alpha_dequantized, output_scale, output_zero_point, input_zero_point
    )

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data, alpha_data
    )

    np.testing.assert_equal(op_res.numpy(), golden_output)


if __name__ == "__main__":
    test_qnn_prelu()
