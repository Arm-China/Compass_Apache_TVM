# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
import tvm
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass.transform.hint_pattern_rewrite import (
    RewriteSearchSorted,
)
import tvm.testing


# Pytest Specific Function
def test_pass_rewrite_searchsorted():
    sorted_data = np.random.rand(100)
    sorted_data = np.sort(sorted_data, axis=0)
    sorted_data = sorted_data.astype(np.float32)
    sorted_data = relay.Constant(tvm.nd.array(sorted_data))

    def get_mod(right=True):
        inp = relay.Var("input", tvm.ir.TensorType([300], dtype="float32"))
        out = relay.searchsorted(sorted_data, inp, right)
        mod = tvm.IRModule.from_expr(out)
        mod = relay.transform.InferType()(mod)
        return mod

    mod = get_mod(True)
    input_val = np.random.rand(200)
    input_val = input_val.astype(np.float32)

    input_val = np.concatenate([input_val, sorted_data.data.numpy()], axis=0)
    np.random.shuffle(input_val)
    gt_val = relay.create_executor(mod=mod).evaluate()(input_val)
    update_mod = RewriteSearchSorted()(mod)
    val = relay.create_executor(mod=update_mod).evaluate()(input_val)
    tvm.testing.assert_allclose(val.numpy(), gt_val.numpy(), atol=0, rtol=0)

    mod = get_mod(False)
    gt_val = relay.create_executor(mod=mod).evaluate()(input_val)
    update_mod = RewriteSearchSorted()(mod)
    val = relay.create_executor(mod=update_mod).evaluate()(input_val)
    tvm.testing.assert_allclose(val.numpy(), gt_val.numpy(), atol=0, rtol=0)


if __name__ == "__main__":
    test_pass_rewrite_searchsorted()
