# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
import tvm
from tvm import relax
from tvm.compass.relax import Compass


@pytest.mark.xfail(reason="CP-15684")
def test_specified_partition():
    inp0 = relax.var("x0", shape=[1, 224, 224, 3], dtype="float32")
    weight_datas = np.random.random([32, 3, 3, 3])
    conv = relax.nn.conv2d(
        inp0,
        relax.const(weight_datas, dtype="float32"),
        channels=32,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="OHWI",
    )
    add = relax.add(conv, relax.const(np.ones([1, 1, 1, 32]), dtype="float32"))
    relu = relax.nn.relu(add)
    out = relax.nn.softmax(relu)
    func = relax.Function([inp0], out)
    mod = tvm.IRModule.from_expr(func)
    mod = relax.transform.InferType()(mod)

    cfg_str = """
    [Common]
    dump_annotation_graph = True

    [Parser]
    model_type = relax
    model_name = relax_test
    """
    compass = Compass(cfg_str)
    compass.ir_mod = mod
    compass.optimize()
    compass.partition(fallback_indices=[7])
    main_func_text = compass.ir_mod["main"].astext()

    expect = """\
#[version = "0.0.5"]
fn (%x0: Tensor[(1, 224, 224, 3), float32] /* ty=Tensor[(1, 224, 224, 3), float32] */) -> Tensor[(1, 222, 222, 32), float32] {
  %0 = @tvm_compass_main_0(%x0) /* ty=Tensor[(1, 222, 222, 32), float32] */;
  nn.softmax(%0) /* ty=Tensor[(1, 222, 222, 32), float32] */
} /* ty=fn (Tensor[(1, 224, 224, 3), float32]) -> Tensor[(1, 222, 222, 32), float32] */
"""
    assert main_func_text == expect.strip(), f"\nExpect snippet:\n{expect}\n\nActual snippet:\n{main_func_text}\n"

    compass.ir_mod = mod
    compass.optimize()
    compass.partition(fallback_indices=[4])
    compass.build()


if __name__ == "__main__":
    test_specified_partition()
