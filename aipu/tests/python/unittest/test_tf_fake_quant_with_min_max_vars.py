# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import tensorflow as tf
import numpy as np
import tvm
from tvm import relay
from tvm.relay.op.contrib import aipu_compass


@pytest.mark.parametrize(
    "input_shapes",
    [
        [[2, 3]],
        [[2, 3, 4]],
        [[10, 20, 12, 12]],
    ],
)
@pytest.mark.parametrize(
    "min_max",
    [[-6, 6], [1.23, 4.56], [-0.789, -0.123]],
)
@pytest.mark.parametrize(
    "narrow_range",
    [False, True],
)
@pytest.mark.parametrize(
    "num_bits",
    [2, 8, 16],
)
def test_fake_quant_with_min_max_vars(input_shapes, min_max, narrow_range, num_bits):
    np.random.seed(0)
    inp = np.random.random(input_shapes[0])
    inp = inp.astype("float32")
    tf_out = tf.quantization.fake_quant_with_min_max_vars(inp, min_max[0], min_max[1], num_bits, narrow_range).numpy()
    relay_inp = relay.const(inp)
    relay_out = aipu_compass.fake_quant_with_min_max_vars(relay_inp, min_max[0], min_max[1], narrow_range, num_bits)
    mod = tvm.IRModule.from_expr(relay_out)
    tvm_out = relay.create_executor(mod=mod, device=tvm.cpu(0), target="llvm").evaluate()().numpy()
    delta = np.max(np.abs(tvm_out - tf_out))
    assert delta < 10e-4
