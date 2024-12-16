# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
from tvm.relay.backend.contrib.aipu_compass import AipuCompass
from tvm.relay.backend.contrib.aipu_compass import testing as aipu_testing


def test_bypass_dequant():
    cfg_path = f"{aipu_testing.DATA_DIR}/relay_tiny.cfg"

    # 1. Normal compilation. Raw model's output dtype is float32 different with
    # compass output uint8. So qnn.dequantize will be inserted automatically.
    compass = AipuCompass(cfg_path)
    compass.compile()
    # print(compass.ir_mod["main"].astext())
    # #[version = "0.0.5"]
    # fn (%input: Tensor[(1, 224, 224, 3), float32] /* ty=Tensor[(1, 224, 224, 3), float32] span=from_string:3:18 */) -> Tensor[(1, 56, 56, 256), float32] {
    #   %0 = qnn.quantize(%input, 1.18431f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8") /* ty=Tensor[(1, 224, 224, 3), int8] */;
    #   %1 = @tvmgen_default_aipu_compass_main_0(%0) /* ty=Tensor[(1, 56, 56, 256), int8] */;
    #   qnn.dequantize(%1, 0.171154f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="float32") /* ty=Tensor[(1, 56, 56, 256), float32] */
    # } /* ty=fn (Tensor[(1, 224, 224, 3), float32]) -> Tensor[(1, 56, 56, 256), float32] */

    # 2. If don't need dequantize, bypass it as follows.
    # bypass_output_dequant indicates list of compass output indices.
    compass1 = AipuCompass(cfg_path)
    bypass_output_dequant = [0]
    compass1.compile(bypass_output_dequant=bypass_output_dequant)
    bypass_dequant = compass1.ir_mod["main"].astext()

    expect = """\
#[version = "0.0.5"]
fn (%input: Tensor[(1, 224, 224, 3), float32] /* ty=Tensor[(1, 224, 224, 3), float32] span=from_string:3:18 */) -> Tensor[(1, 56, 56, 256), int8] {
  %0 = qnn.quantize(%input, 1.18431f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8") /* ty=Tensor[(1, 224, 224, 3), int8] */;
  @tvmgen_default_aipu_compass_main_0(%0) /* ty=Tensor[(1, 56, 56, 256), int8] */
} /* ty=fn (Tensor[(1, 224, 224, 3), float32]) -> Tensor[(1, 56, 56, 256), int8] */
"""
    assert bypass_dequant == expect.strip(), f"\nExpect snippet:\n{expect}\n\nActual snippet:\n{bypass_dequant}\n"


if __name__ == "__main__":
    test_bypass_dequant()
