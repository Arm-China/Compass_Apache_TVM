# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
import numpy as np
import tvm
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass import AipuCompass


def test_specified_partition():
    inp0 = relay.var("x0", shape=[1, 224, 224, 3], dtype="float32")
    weight_datas = np.random.random([32, 3, 3, 3])
    conv = relay.nn.conv2d(
        inp0,
        relay.const(weight_datas, dtype="float32"),
        channels=32,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="OHWI",
    )
    add = relay.add(conv, relay.const(np.ones([1, 1, 1, 32]), dtype="float32"))
    relu = relay.nn.relu(add)
    out = relay.nn.softmax(relu)
    func = relay.Function([inp0], out)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    cfg_str = """
    [Common]
    dump_annotation_graph = True

    [Parser]
    model_type = relay
    model_name = relay_test
    """
    compass = AipuCompass(cfg_str)
    compass.ir_mod = mod
    compass.optimize()
    compass.partition(fallback_indices=[7])
    main_func_text = compass.ir_mod["main"].astext()

    expect_snippet = """\
#[version = "0.0.5"]
fn (%x0: Tensor[(1, 224, 224, 3), float32] /* ty=Tensor[(1, 224, 224, 3), float32] */) -> Tensor[(1, 222, 222, 32), float32] {
  %0 = @tvmgen_default_aipu_compass_main_0(%x0) /* ty=Tensor[(1, 222, 222, 32), float32] */;
  nn.softmax(%0) /* ty=Tensor[(1, 222, 222, 32), float32] */
} /* ty=fn (Tensor[(1, 224, 224, 3), float32]) -> Tensor[(1, 222, 222, 32), float32] */
"""
    assert (
        main_func_text == expect_snippet.strip()
    ), f"\nExpect snippet:\n{expect_snippet}\n\nActual snippet:\n{main_func_text}\n"

    compass.ir_mod = mod
    compass.optimize()
    compass.partition(fallback_indices=[4])
    compass.build()


if __name__ == "__main__":
    test_specified_partition()
