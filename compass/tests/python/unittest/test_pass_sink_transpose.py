# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import os
import onnx
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.compass.relax import transform as compass_transform


def _get_op_freqs(expr, op_name):
    ret = 0

    def fvisit(x):
        nonlocal ret
        if isinstance(x, relax.Call) and x.op.name == op_name:
            ret += 1

    relax.analysis.post_order_visit(expr, fvisit)
    return ret


def test_sink_transpose():
    model = onnx.load(f"{os.getenv('ZHOUYI_MODEL_ZOO_HOME')}/onnx_resnet_v1_50/resnet_v1_50.onnx")
    shape_dict = {"import/Placeholder:0": (1, 3, 224, 224)}

    ir_mod = from_onnx(model, shape_dict=shape_dict)
    mod = compass_transform.SinkTranspose()(ir_mod)
    print(mod)
    assert _get_op_freqs(mod["main"], "relax.permute_dims") == 1


if __name__ == "__main__":
    test_sink_transpose()
