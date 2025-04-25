# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Utilities for pattern table implementation."""
import numpy as np
from tvm import relay


def get_activation_str(output_scale, output_zp, clip):
    """Get the AIPU Compass "with_activation" value from Relay IR."""
    output_scale = float(output_scale.data.numpy())
    output_zp = int(output_zp.data.numpy())
    # Dequantize a quantized integer value to a float value.
    dequantize = lambda x: output_scale * (x - output_zp)
    min_value = dequantize(clip.attrs.a_min)
    max_value = dequantize(clip.attrs.a_max)
    if np.isclose(min_value, 0):
        if np.isclose(max_value, 6, 1.0e-2):
            return "RELU6"
        return "RELU"
    return None


def unpack_commutative_args(call, rhs_name="const"):
    """Unpack arguments of the binary operators consider commutative, ensure the
    right hand side operand is the expected one."""
    assert isinstance(call, relay.Call)
    ops = (
        relay.op.get("add"),
        relay.op.get("multiply"),
        relay.op.get("maximum"),
        relay.op.get("minimum"),
    )

    lhs, rhs = call.args
    if rhs_name == "const":
        if isinstance(rhs, relay.Constant):
            return lhs, rhs
        assert call.op in ops
        assert isinstance(lhs, relay.Constant)
        return rhs, lhs

    if isinstance(rhs, relay.Call) and rhs.op == relay.op.get(rhs_name):
        return lhs, rhs

    assert call.op in ops
    assert isinstance(lhs, relay.Call) and lhs.op == relay.op.get(rhs_name)
    return rhs, lhs


def is_scalar_and_close(x, ref):
    """Check if the given argument is a scalar and close to the given reference value."""
    assert isinstance(x, (relay.Constant, int, float))
    assert isinstance(ref, (int, float))
    if isinstance(x, relay.Constant):
        x = x.data.numpy()
    if x.size != 1:
        return False
    return np.isclose(float(x), float(ref))


def peel_hardswish(hardswish):
    """Peel the needed nodes out."""
    args = hardswish.args
    if any(isinstance(arg, relay.Constant) for arg in args):
        # The last OP is multiply 1/6 or divide 6.
        # (clip(x + 3) * x) / 6 or (clip(x + 3) * x) * 1/6
        six_mul_or_div = hardswish
        add_clip_mul, _ = unpack_commutative_args(six_mul_or_div)
        data, clip = unpack_commutative_args(add_clip_mul, "clip")
    else:
        # The last OP is the other multiply.
        if any(isinstance(arg, relay.Call) and arg.op == relay.op.get("clip") for arg in args):
            # x / 6 * clip(x + 3) or x * 1/6 * clip(x + 3)
            six_mul_or_div, clip = unpack_commutative_args(hardswish, "clip")
        else:
            # clip(x + 3) / 6 * x or clip(x + 3) * 1/6 * x
            six_mul_or_div = [
                arg
                for arg in args
                if isinstance(arg, relay.Call)
                and arg.op in (relay.op.get("multiply"), relay.op.get("divide"))
            ]

            def _check_six(call):
                if all([not isinstance(arg, relay.Constant) for arg in call.args]):
                    return False
                _, const_v = unpack_commutative_args(call)
                if call.op == relay.op.get("multiply") and is_scalar_and_close(const_v, 1 / 6):
                    return True
                if call.op == relay.op.get("divide") and is_scalar_and_close(const_v, 6):
                    return True
                return False

            six_mul_or_div = [expr for expr in six_mul_or_div if _check_six(expr)][0]
            clip, _ = unpack_commutative_args(six_mul_or_div)
        data, _ = unpack_commutative_args(clip.args[0])
    return data, six_mul_or_div, clip
