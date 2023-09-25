# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
# pylint: disable=unused-argument
"""Strategy of AIPU Compass extended operators."""
import cython
from tvm import te, tir, topi
from tvm.target import override_native_generic_func
from ...op import register_strategy, OpStrategy
from ... import strategy
from .....topi.nn.utils import get_pad_tuple
from .....topi.utils import get_const_tuple
from .....topi.cpp.utils import bilinear_sample_nhwc


def matrix_band_part_compute(attrs, inputs, out_dtype):
    """The compute of MatrixBandPart operator."""
    num_lower = attrs.num_lower
    num_upper = attrs.num_upper
    shape = inputs[0].shape
    inp = inputs[0]

    def compute(*shapes):
        *_, mdim, ndim = shapes
        lower_cond = tir.any(num_lower < tir.const(0, "int32"), (mdim - ndim) <= num_lower)
        upper_cond = tir.any(num_upper < tir.const(0, "int32"), (ndim - mdim) <= num_upper)
        cond = tir.all(lower_cond, upper_cond)
        return te.if_then_else(cond, inp(*shapes), tir.const(0, out_dtype.dtype))

    return [te.compute(shape, compute)]


@override_native_generic_func("contrib.aipu_compass.matrix_band_part")
@cython.binding(True)
def matrix_band_part_strategy(attrs, inputs, out_type, target):
    """The generic strategy of MatrixBandPart operator."""

    _strategy = OpStrategy()
    _strategy.add_implementation(
        matrix_band_part_compute,
        strategy.naive_schedule,
        name="contrib.aipu_compass.matrix_band_part.generic",
    )
    return _strategy


def fake_quant_with_min_max_vars_compute(attrs, inputs, _):
    """The compute of FakeQuantWithMinMaxVars operator."""
    inp = inputs[0]
    narrow_range = attrs.narrow_range
    num_bits = attrs.num_bits
    minimum = attrs.minimum
    maximum = attrs.maximum

    if minimum > 0:
        min_adj = 0
        max_adj = maximum - minimum
    elif maximum < 0:
        min_adj = minimum - maximum
        max_adj = 0
    else:
        scale = (maximum - minimum) / ((1 << num_bits) - 1)
        min_adj = scale * round(minimum / scale)
        max_adj = maximum + min_adj - minimum

    if narrow_range:
        qmin = 1
    else:
        qmin = 0
    qmax = (1 << num_bits) - 1
    scale = (qmax - qmin) / (max_adj - min_adj)

    return [topi.clip((topi.clip(topi.round(scale * inp), 0, qmax) / scale), min_adj, max_adj)]


@override_native_generic_func("contrib.aipu_compass.fake_quant_with_min_max_vars")
@cython.binding(True)
def fake_quant_with_min_max_vars_strategy(attrs, inputs, out_type, target):
    """The generic strategy of FakeQuantWithMinMaxVars operator."""

    _strategy = OpStrategy()
    _strategy.add_implementation(
        fake_quant_with_min_max_vars_compute,
        strategy.naive_schedule,
        name="contrib.aipu_compass.fake_quant_with_min_max_vars.generic",
    )

    return _strategy


def compute_deformable_conv2d_v2_nhwc(attrs, inputs, out_dtype):
    """wrap deformable_conv2d topi compute"""

    padding = get_const_tuple(attrs.padding)
    strides = get_const_tuple(attrs.strides)
    dilation = get_const_tuple(attrs.dilation)
    deformable_groups = attrs.deformable_groups
    groups = attrs.groups
    out_dtype = attrs.out_dtype
    out_dtype = inputs[0].dtype if out_dtype in ("same", "") else out_dtype

    data = inputs[0]
    offset = inputs[1]
    kernel = inputs[2]
    mask = inputs[3]

    if out_dtype is None:
        out_dtype = data.dtype

    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_height, in_width, in_channel = get_const_tuple(data.shape)
    kernel_h, kernel_w, channel, out_channel = get_const_tuple(kernel.shape)
    _, out_height, out_width, _ = get_const_tuple(offset.shape)

    assert in_channel % deformable_groups == 0, "Input channels must divide deformable group size"
    assert groups == 1, "deformable_conv2d_nchw does not support groups > 1"

    ic_per_dgroup = channel // deformable_groups

    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, _, _ = get_pad_tuple(padding, (dilated_kernel_h, dilated_kernel_w))

    zero = tir.const(0.0, data.dtype)

    def _bilinear(rn_, rh_, rw_, rc_):
        outside = tir.any(rh_ < 0, rw_ < 0, rh_ >= in_height, rw_ >= in_width)
        val = bilinear_sample_nhwc(data, (rn_, rh_, rw_, rc_), in_height - 1, in_width - 1)
        return tir.if_then_else(outside, zero, val)

    data_deform = te.compute(
        (batch, kernel_h, kernel_w, in_channel, out_height, out_width),
        lambda n, kh, kw, c, y, x: _bilinear(
            n,
            y * stride_h
            - pad_top
            + kh * dilation_h
            + offset[
                n, y, x, c // ic_per_dgroup * (kernel_w * kernel_h * 2) + (kh * kernel_w + kw) * 2
            ],
            x * stride_w
            - pad_left
            + kw * dilation_w
            + offset[
                n,
                y,
                x,
                c // ic_per_dgroup * (kernel_w * kernel_h * 2) + (kh * kernel_w + kw) * 2 + 1,
            ],
            c,
        ),
        tag="data_deform",
    )

    rc_ = te.reduce_axis((0, in_channel), name="rc")
    ry_ = te.reduce_axis((0, kernel_h), name="ry")
    rx_ = te.reduce_axis((0, kernel_w), name="rx")
    out = te.compute(
        (batch, out_height, out_width, out_channel),
        lambda n, y, x, f: te.sum(
            data_deform[n, ry_, rx_, rc_, y, x].astype(out_dtype)
            * kernel[ry_, rx_, rc_, f].astype(out_dtype)
            * mask[
                n, y, x, rc_ // ic_per_dgroup * kernel_w * kernel_h + ry_ * kernel_w + rx_
            ].astype(out_dtype),
            axis=[ry_, rx_, rc_],
        ),
        tag="deformable_conv2d_v2_nhwc",
    )
    return [out]


@override_native_generic_func("deformable_conv2d_v2_strategy")
@cython.binding(True)
def deformable_conv2d_v2_strategy(attrs, inputs, out_type, target):
    """deformable_conv2d generic strategy"""
    layout = attrs.data_layout
    _strategy = OpStrategy()

    if layout == "NHWC":
        # This implementation should never be picked by autotvm
        _strategy.add_implementation(
            compute_deformable_conv2d_v2_nhwc,
            strategy.naive_schedule,
            name="deformable_conv2d_v2_nhwc.generic",
        )
    else:
        raise RuntimeError("Layout %s is not supported in deformable conv2d" % layout)
    return _strategy


register_strategy("contrib.aipu_compass.matrix_band_part", matrix_band_part_strategy)
register_strategy(
    "contrib.aipu_compass.fake_quant_with_min_max_vars", fake_quant_with_min_max_vars_strategy
)
register_strategy("contrib.aipu_compass.deformable_conv2d_v2", deformable_conv2d_v2_strategy)
