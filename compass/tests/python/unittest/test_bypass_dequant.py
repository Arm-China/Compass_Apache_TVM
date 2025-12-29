# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
from tvm.compass.relax import Compass, testing


def test_bypass_output_dequant():
    cfg_path = f"{testing.DATA_DIR}/relax_tiny.cfg"

    # 1. Normal compilation. Raw model's output dtype is float32 different with
    # compass output uint8. So qnn.dequantize will be inserted automatically.
    # compass = Compass(cfg_path)
    # compass.compile()
    # print(compass.ir_mod["main"].script())
    # # from tvm.script import relax as R

    # @R.function
    # def main(input: R.Tensor((1, 224, 224, 3), dtype="float32")) -> R.Tensor((1, 56, 56, 256), dtype="float32"):
    #     R.func_attr({"num_input": 1})
    #     with R.dataflow():
    #         lv: R.Tensor((1, 224, 224, 3), dtype="int8") = R.quantize(input, R.const(1.1843137741088867, "float32"), R.const(0, "int32"), out_dtype="int8", axis=-1)
    #         lv1 = R.call_dps_packed("tvm_compass_subfunc0", (lv,), out_sinfo=R.Tensor((1, 56, 56, 256), dtype="int8"))
    #         gv: R.Tensor((1, 56, 56, 256), dtype="float32") = R.dequantize(lv1, R.const(0.12785191833972931, "float32"), R.const(0, "int32"), out_dtype="float32", axis=-1)
    #         R.output(gv)
    #     return gv

    # 2. If don't need dequantize, bypass it as follows.
    # bypass_output_dequant indicates list of compass output indices.
    compass1 = Compass(cfg_path)
    compass1.compile(bypass_output_dequant=[0])
    bypass_dequant = compass1.ir_mod["main"].script()

    expect = """\
# from tvm.script import relax as R

@R.function
def main(input: R.Tensor((1, 224, 224, 3), dtype="float32")) -> R.Tensor((1, 56, 56, 256), dtype="int8"):
    R.func_attr({"num_input": 1})
    with R.dataflow():
        lv: R.Tensor((1, 224, 224, 3), dtype="int8") = R.quantize(input, R.const(1.1843137741088867, "float32"), R.const(0, "int32"), out_dtype="int8", axis=-1)
        gv = R.call_dps_packed("tvm_compass_subfunc0", (lv,), out_sinfo=R.Tensor((1, 56, 56, 256), dtype="int8"))
        R.output(gv)
    return gv
"""
    assert bypass_dequant in expect, f"\nExpect snippet:\n{expect}\n\nActual snippet:\n{bypass_dequant}\n"


if __name__ == "__main__":
    test_bypass_output_dequant()
