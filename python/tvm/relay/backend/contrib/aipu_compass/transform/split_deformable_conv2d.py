# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
"""split deformable_conv2d operator."""

from tvm import relay
import numpy as np


def gen_offset_base(in_h: int, in_w: int, o_h: int, o_w: int, k_h: int, k_w: int, strides):
    """anchor"""
    strides = (int(strides[0]), int(strides[1]))
    start_w = int((k_w - 1) // 2)
    start_h = int((k_h - 1) // 2)
    o_h = int(o_h)
    o_w = int(o_w)
    base_anchor_x = np.linspace(-1.0, 1.0, num=int(in_w), endpoint=True)
    base_anchor_x = base_anchor_x[start_w :: strides[1]][0 : int(o_w)]
    base_anchor_x = base_anchor_x.reshape(1, 1, 1, -1)

    base_anchor_y = np.linspace(-1.0, 1.0, num=int(in_h), endpoint=True)
    base_anchor_y = base_anchor_y[start_h :: strides[0]][0 : int(o_h)]
    base_anchor_y = base_anchor_y.reshape(1, 1, -1, 1)

    base_anchor_x = np.tile(base_anchor_x, (1, 1, o_h, 1))
    base_anchor_y = np.tile(base_anchor_y, (1, 1, 1, o_w))

    base_anchor = np.concatenate([base_anchor_x, base_anchor_y], axis=1)
    base_anchor = np.tile(base_anchor, (int(k_h * k_w), 1, 1, 1))

    step_w = 2.0 / (in_w - 1)  # align corner
    step_h = 2.0 / (in_h - 1)
    start_kernel_rel_offset_x = -((k_w - 1) // 2)
    start_kernel_rel_offset_y = -((k_h - 1) // 2)
    end_kernel_rel_offset_x = k_w - 1 + start_kernel_rel_offset_x
    end_kernel_rel_offset_y = k_h - 1 + start_kernel_rel_offset_y

    kernel_x_offset = np.tile(
        np.linspace(
            float(start_kernel_rel_offset_x * step_w),
            float(end_kernel_rel_offset_x * step_w),
            num=int(k_w),
            endpoint=True,
        ),
        int(k_h),
    ).reshape(int(k_h * k_w))
    kernel_y_offset = np.repeat(
        np.linspace(
            start_kernel_rel_offset_y * step_h,
            end_kernel_rel_offset_y * step_h,
            num=int(k_h),
            endpoint=True,
        ),
        int(k_w),
    ).reshape(int(k_h * k_w))

    # kx00,ky00,kx01,ky01...kx22,ky22
    kernel_offset = np.concatenate(
        (kernel_x_offset[:, np.newaxis], kernel_y_offset[:, np.newaxis]), axis=1
    )

    kernel_offset = kernel_offset.reshape((int(k_h * k_w), 2, 1, 1))
    kernel_offset = np.tile(kernel_offset, [1, 1, o_h, o_w])
    base = base_anchor + kernel_offset

    return base.astype(np.float32)


def grid_sample_conv(
    data,
    offset,
    weight,
    mask,
    base,
    batch,
    in_c,
    in_h,
    in_w,
    out_c,
    o_h,
    o_w,
    k_h,
    k_w,
    layout="NHWC",
):
    """defomable conv to grid sample + conv2d"""

    if layout == "NHWC":
        data = relay.transpose(data, [0, 3, 1, 2])  # NHWC to NCHW
        offset = relay.transpose(offset, [0, 3, 1, 2])
    data = relay.tile(data, [k_h * k_w, 1, 1, 1])
    offset = relay.reshape(offset, [batch * k_h * k_w, 2, o_h, o_w])

    split0, split1 = relay.split(offset, 2, axis=1)  # yx to xy
    split0 = relay.multiply(split0, relay.const(float(2.0 / (in_h - 1)), dtype="float32"))
    split1 = relay.multiply(split1, relay.const(float(2.0 / (in_w - 1)), dtype="float32"))
    offset = relay.concatenate([split1, split0], axis=1)
    offset = relay.add(offset, base)
    data = relay.transpose(data, [0, 2, 3, 1])
    offset = relay.transpose(offset, [0, 2, 3, 1])
    data = relay.image.grid_sample(data, offset, method="bilinear", layout="NHWC")
    if mask is not None:
        if layout == "NHWC":
            mask = relay.transpose(mask, [0, 3, 1, 2])
            mask = relay.reshape(mask, [batch * k_h * k_w, 1, o_h, o_w])
            mask = relay.transpose(mask, [0, 2, 3, 1])
        else:
            mask = relay.reshape(mask, [batch * k_h * k_w, 1, o_h, o_w])
        data = data * mask
    data = relay.transpose(data, [1, 2, 0, 3])
    data = relay.reshape(data, [batch, o_h, o_w, k_h * k_w * in_c])

    # HWIO
    weight = relay.const(
        weight.data.numpy()
        .transpose([3, 0, 1, 2])
        .reshape([int(out_c), 1, 1, int(k_h * k_w * in_c)])
    )
    expr = relay.op.nn.conv2d(
        data,
        weight,
        strides=(1, 1),
        padding=(0, 0),
        kernel_size=(1, 1),
        channels=out_c,
        data_layout="NHWC",
        kernel_layout="OHWI",
    )
    return expr


class DeformableConv2dSpliter(relay.ExprMutator):
    """Split operator "deformable_conv2d"."""

    def visit_call(self, call):
        """
        1. Check
        2. analysis params
        3. make args
        4. make expr
        """
        ret = super().visit_call(call)

        deformable_conv = [
            relay.op.get("nn.deformable_conv2d"),
            relay.op.get("contrib.aipu_compass.deformable_conv2d_v2"),
        ]
        if ret.op not in deformable_conv:
            return ret

        data, offset, weight, mask = None, None, None, None
        if len(ret.args) == 3:
            data, offset, weight = ret.args
        else:
            data, offset, weight, mask = ret.args
        dshape = call.args[0].checked_type.shape
        k_shape = call.args[2].checked_type.shape
        strides = ret.attrs["strides"]

        batch, in_h, in_w, in_c = dshape  # NHWC
        k_h, k_w, in_c, out_c = k_shape  # HWIO
        o_h = (in_h - k_h) // strides[0] + 1
        o_w = (in_w - k_w) // strides[1] + 1
        assert call.args[1].checked_type.shape[1] == o_h, "offset shape H and params conflict."
        assert call.args[1].checked_type.shape[2] == o_w, "offset shape W and params conflict."
        base = gen_offset_base(in_h, in_w, o_h, o_w, k_h, k_w, strides)
        base = relay.const(base, dtype="float32")
        ret = grid_sample_conv(
            data,
            offset,
            weight,
            mask,
            base,
            batch,
            in_c,
            in_h,
            in_w,
            out_c,
            o_h,
            o_w,
            k_h,
            k_w,
        )

        return ret


@relay.transform.function_pass(opt_level=0)
class SplitDeformableConv2d:
    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        return DeformableConv2dSpliter().visit(func)
