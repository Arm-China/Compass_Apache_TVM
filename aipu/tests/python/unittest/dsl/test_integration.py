# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
import tvm
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass import AipuCompass
from tvm.aipu.utils import rand

CFG = """
[Common]
auto_fuse_ops = true

[Parser]

[Optimizer]
calibration_batch_size = 1
metric_batch_size = 1
weight_bits = 8
bias_bits = 32
activation_bits = 8
dataset = NumpyDataset
calibration_data = dataset.npy

[GBuilder]
target = X2_1204
"""


def compute_cos_distance(x, y):
    """Get cosine similarity."""
    x = x.astype("float32")
    y = y.astype("float32")
    similarity = np.dot(x.flatten(), y.flatten()) / (np.linalg.norm(x) * (np.linalg.norm(y)))
    return float(format(similarity, ".3f"))


def test_integration():
    shape = [1, 10, 10, 32]
    dtype = "float32"
    add_data = rand(shape, dtype, 1, 2, enable_corner_values=False)
    sub_data = rand(shape, dtype, enable_corner_values=False)
    add1_data = rand(shape, dtype, 1, 2, enable_corner_values=False)
    sub1_data = rand(shape, dtype, enable_corner_values=False)
    dataset = rand(shape, dtype, enable_corner_values=False)
    np.save("dataset.npy", dataset)

    inp = relay.var("x", shape=shape, dtype=dtype)
    add = relay.add(inp, relay.const(add_data, dtype=dtype))
    sub = relay.subtract(add, relay.const(sub_data, dtype=dtype))
    add1 = relay.add(sub, relay.const(add1_data, dtype=dtype))
    sub1 = relay.subtract(add1, relay.const(sub1_data, dtype=dtype))

    func = relay.Function([inp], sub1)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    # 1. Create AIPU Compass instance and set configurations.
    compass = AipuCompass(CFG)

    # 2. Compile the nn model.
    compass.ir_mod = mod
    compass.optimize()
    compass.partition()
    compass.collect_calibration_data()
    deployable = compass.build(target="llvm")

    # 3. Create execution engine.
    rpc_sess, device_compiler = None, None
    ee = deployable.create_execution_engine(rpc_sess, cc=device_compiler)

    # 4. Run the nn model.
    image = dataset
    outputs = ee.run(image)
    gt_data = image + add_data - sub_data + add1_data - sub1_data
    cosin = compute_cos_distance(gt_data, outputs[0].numpy())
    print(cosin)
    assert cosin >= 0.97


if __name__ == "__main__":
    test_integration()
