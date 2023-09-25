# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
import pytest
import tensorflow as tf
import numpy as np
import tvm
from tvm import relay
from tvm.relay.op.contrib.aipu_compass import matrix_band_part


@pytest.mark.parametrize(
    "input_shapes",
    [
        [[7, 7]],
        [[10, 7, 7]],
        [[10, 20, 7, 7]],
    ],
)
@pytest.mark.parametrize(
    "lower_num",
    [-1, 0, 1],
)
@pytest.mark.parametrize(
    "upper_num",
    [-1, 0, 1],
)
def test_matrix_band_part(input_shapes, lower_num, upper_num):
    inp = np.random.random(input_shapes[0])
    inp = inp.astype("float32")
    tf_out = tf.linalg.band_part(inp, lower_num, upper_num).numpy()

    relay_inp = relay.const(inp)
    relay_out = matrix_band_part(relay_inp, lower_num, upper_num)
    mod = tvm.IRModule.from_expr(relay_out)
    tvm_out = relay.create_executor(mod=mod, device=tvm.cpu(0), target="llvm").evaluate()().numpy()
    delta = np.max(np.abs(tvm_out - tf_out))
    assert delta < 10e-4
