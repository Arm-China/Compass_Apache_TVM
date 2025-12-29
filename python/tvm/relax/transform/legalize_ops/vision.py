# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
# pylint: disable=invalid-name
"""Default legalization function for vision network related operators."""
from tvm import topi
from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from .common import register_legalize


@register_legalize("relax.vision.all_class_non_max_suppression")
def _vision_all_class_non_max_suppression(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        topi.vision.all_class_non_max_suppression,
        call.args[0],
        call.args[1],
        call.args[2],
        call.args[3],
        call.args[4],
        output_format=call.attrs.output_format,
    )
