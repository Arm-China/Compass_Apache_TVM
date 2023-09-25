# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
import os
import onnx
import tvm
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass import transform as compass_transform


# Pytest Specific Function
def test_sink_transpose():
    model = onnx.load(f"{os.getenv('ZHOUYI_MODEL_ZOO_HOME')}/onnx_resnet_v1_50/resnet_v1_50.onnx")
    shape_dict = {"import/Placeholder:0": (1, 3, 224, 224)}

    ir_mod, params = relay.frontend.from_onnx(model, shape=shape_dict)
    ir_mod["main"] = relay.build_module.bind_params_by_name(ir_mod["main"], params)

    # Simplify and optimization.
    passes = [
        relay.transform.RemoveUnusedFunctions(),
        relay.transform.SimplifyInference(),
        relay.transform.DynamicToStatic(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.SimplifyExpr(),
        compass_transform.SinkTranspose(),
        relay.transform.SimplifyExpr(),
        relay.transform.FoldConstant(),
        relay.transform.FoldScaleAxis(),
        relay.transform.CanonicalizeOps(),
        compass_transform.SimplifyPad(),
    ]
    with tvm.transform.PassContext(opt_level=3):
        ir_mod = tvm.transform.Sequential(passes)(ir_mod)

    op_freqs = relay.analysis.list_op_freqs(ir_mod)
    assert op_freqs["transpose"] == 1


if __name__ == "__main__":
    test_sink_transpose()
