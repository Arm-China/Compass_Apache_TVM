# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
"""Simplify operator "pad"."""
from tvm import ir, relay, tir


class PadSimplifier(relay.ExprMutator):
    """Simplifier of operator "pad"."""

    def visit_call(self, call):
        call = super().visit_call(call)

        if (
            call.op != relay.op.get("nn.conv2d")
            or not isinstance(call.args[0], relay.Call)
            or call.args[0].op != relay.op.get("nn.pad")
        ):
            return call

        conv2d = call
        pad, weight = conv2d.args
        data, pad_value = pad.args
        if (
            pad.attrs.pad_mode != "constant"
            or not isinstance(pad_value, relay.Constant)
            or pad_value.data.numpy() != 0
        ):
            # "nn.conv2d" only support padding the constant value "0".
            return conv2d

        # Although "nn.pad" hasn't layout information, but the attribute
        # "data_layout" of "nn.conv2d" can be used to find the indices of
        # dimension "H" and "W".
        data_layout = tir.layout(conv2d.attrs.data_layout)
        if any(x in data_layout for x in ["h", "w"]):
            # The dimension "H" or "W" is split, and "nn.conv2d" can't support
            # padding on these complicated layout data.
            return conv2d
        h_idx = data_layout.index_of("H")
        w_idx = data_layout.index_of("W")

        for i, pad_width in enumerate(pad.attrs.pad_width):
            if i == h_idx:
                pad_top, pad_bottom = pad_width
            elif i == w_idx:
                pad_left, pad_right = pad_width
            elif any(v != 0 for v in pad_width):
                # The operator "nn.pad" has padding on dimensions which are
                # neither "H" nor "W", so it can't be simplified.
                return conv2d

        conv2d_pad = conv2d.attrs.padding
        assert len(conv2d_pad) in [1, 2, 4], f"Exception: {conv2d_pad}"
        if len(conv2d_pad) == 1:
            pad_top += conv2d_pad[0]
            pad_left += conv2d_pad[0]
            pad_bottom += conv2d_pad[0]
            pad_right += conv2d_pad[0]
        elif len(conv2d_pad) == 2:
            pad_top += conv2d_pad[0]
            pad_left += conv2d_pad[1]
            pad_bottom += conv2d_pad[0]
            pad_right += conv2d_pad[1]
        else:
            pad_top += conv2d_pad[0]
            pad_left += conv2d_pad[1]
            pad_bottom += conv2d_pad[2]
            pad_right += conv2d_pad[3]

        new_attrs = {str(k): conv2d.attrs[k] for k in conv2d.attrs.keys()}
        new_attrs["padding"] = (pad_top, pad_left, pad_bottom, pad_right)
        new_attrs = ir.make_node(str(conv2d.attrs).split("(")[0], **new_attrs)

        new_args = [data, weight]
        return relay.Call(conv2d.op, new_args, new_attrs, conv2d.type_args, conv2d.span)


@relay.transform.function_pass(opt_level=0)
class SimplifyPad:
    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        return PadSimplifier().visit(func)
