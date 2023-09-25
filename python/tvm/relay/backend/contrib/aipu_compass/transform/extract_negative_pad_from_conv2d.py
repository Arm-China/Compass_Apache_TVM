# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""Extract negative pad from convolution 2D."""
from tvm import ir, relay, tir


class NegativePadFromConv2dExtractor(relay.ExprMutator):
    """Extract negative pad from convolution 2D."""

    def visit_call(self, call):
        ret = super().visit_call(call)

        if ret.op == relay.op.get("nn.conv2d"):
            conv2d = ret
            conv2d_pad = conv2d.attrs.padding
            assert len(conv2d_pad) == 4
            if all(x >= 0 for x in conv2d_pad):
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
            pad_top, pad_left, pad_bottom, pad_right = conv2d_pad
            pad_width = []
            data_shape = call.args[0].checked_type.shape
            for i in range(len(data_shape)):
                if i == h_idx:
                    pad_before, pad_after = pad_top, pad_bottom
                elif i == w_idx:
                    pad_before, pad_after = pad_left, pad_right
                else:
                    pad_before, pad_after = 0, 0
                pad_width.append((pad_before, pad_after))
            pad = relay.nn.pad(ret.args[0], pad_width)

            new_attrs = {str(k): conv2d.attrs[k] for k in conv2d.attrs.keys()}
            new_attrs["padding"] = (0, 0, 0, 0)
            new_attrs = ir.make_node(str(conv2d.attrs).split("(")[0], **new_attrs)
            new_args = [pad, ret.args[1]]
            return relay.Call(conv2d.op, new_args, new_attrs, conv2d.type_args, conv2d.span)

        return ret


@relay.transform.function_pass(opt_level=0)
class ExtractNegativePadFromConv2d:
    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        return NegativePadFromConv2dExtractor().visit(func)
