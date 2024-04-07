# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=bad-super-call, unsupported-binary-operation
""" Rewrite module by pattern """
import numpy as np
import tvm
from tvm import ir, relay
from tvm.ir.base import structural_equal
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    is_constant,
    is_op,
    is_tuple_get_item,
    wildcard,
    rewrite,
)
from tvm.relay.op.dyn import _make as _dyn_make


def _is_scalar(const_node):
    if not isinstance(const_node, relay.Constant):
        return False
    if len(const_node.checked_type.shape) == 0:
        return True
    if len(const_node.checked_type.shape) == 1 and const_node.checked_type.shape[0] == 1:
        return True
    return False


class RewriteDecorator:
    """wrapper to wrapper rewrite pattern class"""

    def __init__(self, pattern_cls):
        self.pattern_cls = pattern_cls()

    def __call__(self, before_passes=None, after_passes=None):
        pass0 = before_passes if before_passes else []
        if not isinstance(pass0, list):
            pass0 = [pass0]
        pass1 = after_passes if after_passes else []
        if not isinstance(pass1, list):
            pass1 = [pass1]

        def wrapper(mod):
            for before_pass in pass0:
                mod = before_pass(mod)  # pylint: disable=not-callable
            func = rewrite(self.pattern_cls, mod["main"], mod)
            for key in mod.functions:
                if key.name_hint == "main":
                    mod.update_func(key, func)
            for after_pass in pass1:
                mod = after_pass(mod)  # pylint: disable=not-callable
            return mod

        return wrapper


@RewriteDecorator
class FoldDilatedConv2d(DFPatternCallback):
    """fold space_to_batch + conv2d + batch_to_space to conv2d"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.pattern = is_op("nn.batch_to_space_nd")(
            is_op("nn.conv2d")(is_op("nn.space_to_batch_nd")(wildcard()), is_constant())
        )

    def callback(self, pre, post, node_map):
        b2s = node_map[self.pattern][0]
        conv2d = b2s.args[0]
        s2b = conv2d.args[0]

        s2b_block = np.array(s2b.attrs.block_shape)
        b2s_block = np.array(b2s.attrs.block_shape)
        src_dilation = np.array(conv2d.attrs.dilation)
        src_padding = np.array(conv2d.attrs.padding)
        if (
            not all(s2b_block == b2s_block)
            or not all(src_dilation == 1)
            or not all(src_padding == 0)
        ):
            return post

        s2b_pad = s2b.attrs.paddings
        b2s_crop = b2s.attrs.crops
        pad_h, pad_w = s2b_pad
        crop_h, crop_w = b2s_crop
        pad_top = pad_h[0] - crop_h[0]
        pad_bottom = pad_h[1] - crop_h[1]
        pad_left = pad_w[0] - crop_w[0]
        pad_right = pad_w[1] - crop_w[1]

        new_dilation = [s2b_block[0], s2b_block[1]]
        new_padding = [pad_top, pad_left, pad_bottom, pad_right]
        new_call = relay.nn.conv2d(
            s2b.args[0],
            conv2d.args[1],
            conv2d.attrs.strides,
            new_padding,
            new_dilation,
            conv2d.attrs.groups,
            conv2d.attrs.channels,
            conv2d.attrs.kernel_size,
            conv2d.attrs.data_layout,
            conv2d.attrs.kernel_layout,
        )
        return new_call


@RewriteDecorator
class ShapeOfTake(DFPatternCallback):
    """we can infer some constant value from shapeof + take"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.idx = is_constant()
        self.pattern = is_op("take")(wildcard(), self.idx)

    def callback(self, pre, post, node_map):
        idx = node_map[self.idx][0].data.numpy()
        take = node_map[self.pattern][0]
        if not isinstance(take.args[0], relay.Call) or take.args[0].op != relay.op.get("shape_of"):
            return post

        shapeof = take.args[0]
        input_arg = shapeof.args[0]

        pre_ty = input_arg.checked_type
        take_idx = int(idx)
        value = pre_ty.shape[take_idx]
        if isinstance(value, tvm.tir.Any):
            return post
        value = np.array(int(value), dtype=take.checked_type.dtype)
        return relay.Constant(tvm.nd.array(value))


@RewriteDecorator
class DynamicFullConcatShape(DFPatternCallback):
    """we can infer some constant shape component from concatenate + dyn.full"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.pattern = is_op("dyn.full")(is_constant(), wildcard())

    def callback(self, pre, post, node_map):
        full = node_map[self.pattern][0]
        concat = full.args[1]

        if not isinstance(full.args[1], relay.Call) or full.args[1].op != relay.op.get(
            "concatenate"
        ):
            return post
        fields = concat.args[0]

        full_dtype = full.args[0].checked_type.dtype

        const_idx_shape = []
        non_const_idx_shape = []
        correct_non_const_idx = []
        for idx, val in enumerate(fields):
            if isinstance(val, relay.Constant):
                const_shape = int(val.data.numpy())
                const_idx_shape.append(const_shape)
            else:
                const_idx_shape.append(1)
                non_const_idx_shape.append(val)
                correct_non_const_idx.append(idx)

        if len(non_const_idx_shape) == len(const_idx_shape):
            return post

        const_node = np.zeros(const_idx_shape)
        const_node = const_node.astype(full_dtype)
        const_node = relay.Constant(tvm.nd.array(const_node))

        ret = full.args[0]
        dyn_shape = relay.concatenate(non_const_idx_shape, axis=0)
        ret_dyn = _dyn_make.tile(ret, dyn_shape)
        dims = len(const_idx_shape)
        const_idxes = [idx for idx in range(dims) if idx not in correct_non_const_idx]
        idx0 = correct_non_const_idx + const_idxes
        idx1 = [idx0.index(idx) for idx in range(dims)]

        if len(non_const_idx_shape) != len(const_idx_shape):
            num_newaxis = len(const_idx_shape) - len(non_const_idx_shape)
            ret_dyn = relay.expand_dims(ret_dyn, -1, num_newaxis)

        if idx1 != list(range(dims)):
            ret_dyn = relay.transpose(ret_dyn, idx1)
        return ret_dyn + const_node


@RewriteDecorator
class RecombineSoftmax(DFPatternCallback):
    """Recombine max sub exp sum div to softmax"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        add = is_op("add")(wildcard(), is_constant())
        inp = add | wildcard()
        reduce_max = is_op("max")(inp)
        tile = is_op("tile")(reduce_max)
        sub = is_op("subtract")(inp, tile | reduce_max)
        exp = is_op("exp")(sub | wildcard())
        reduce_sum = is_op("sum")(exp)
        tile = is_op("tile")(reduce_sum)
        self.pattern = is_op("divide")(exp, tile | reduce_sum)

    def callback(self, pre, post, node_map):
        # get input and axis
        div = node_map[self.pattern][0]
        if div.args[1].op == relay.op.get("tile"):
            tile = div.args[1]
            reduce_sum = tile.args[0]
        else:
            reduce_sum = div.args[1]
        exp = reduce_sum.args[0]
        sub = exp.args[0]
        if sub.args[1].op == relay.op.get("tile"):
            tile = sub.args[1]
            reduce_max = tile.args[0]
        else:
            reduce_max = sub.args[1]
        if reduce_max.args[0].op == relay.op.get("add"):
            add = reduce_max.args[0]
            inp = add.args[0]
        else:
            inp = reduce_max.args[0]
        axis = reduce_sum.attrs.axis

        if not isinstance(axis, tvm.ir.container.Array):
            return post

        return relay.nn.softmax(inp, axis=int(axis[0]))


@RewriteDecorator
class Power2ToMul(DFPatternCallback):
    """Remove minimum op for int dtype."""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.pattern = is_op("power")(wildcard(), is_constant())

    def callback(self, pre, post, node_map):
        arg, power = node_map[self.pattern][0].args
        if _is_scalar(power) and float(power.data.numpy()) == 2:
            return arg * arg
        return post


@RewriteDecorator
class EliminateIdentityOp(DFPatternCallback):
    """Eliminates expressions that are equivalent to identity."""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.idx = wildcard()
        reshape = is_op("reshape")(self.idx)
        transpose = is_op("transpose")(self.idx)
        split = is_op("split")(self.idx)
        split = is_tuple_get_item(split, 0)
        resize2d = is_op("image.resize2d")(self.idx)
        concat = is_op("concatenate")(self.idx)
        avg_pool2d = is_op("nn.avg_pool2d")(self.idx)

        self.pattern = reshape | transpose | split | resize2d | concat | avg_pool2d

    def callback(self, pre, post, node_map):
        idx = node_map[self.idx][0]
        call = node_map[self.pattern][0]
        if isinstance(call, relay.Call) and call.op == relay.op.get("transpose"):
            if call.attrs.axes is None:
                return post
            axes = [int(axis) for axis in call.attrs.axes]
            if axes != list(range(len(axes))):
                return post

        if isinstance(call, relay.Call) and call.op == relay.op.get("concatenate"):
            return idx[0] if len(idx) == 1 else post

        if isinstance(call, relay.Call) and call.op == relay.op.get("nn.avg_pool2d"):
            pool_size_set = np.unique(call.attrs.pool_size)
            is_pool1x1 = pool_size_set.size == 1 and pool_size_set == np.array([1])
            padding_set = np.unique(call.attrs.padding)
            is_pad_0 = padding_set.size == 1 and padding_set == np.array([0])
            stride_set = np.unique(call.attrs.strides)
            is_stride_1 = stride_set.size == 1 and stride_set == np.array([1])
            if not (is_pool1x1 and is_pad_0 and is_stride_1):
                return post

        pre_type = pre.checked_type
        x_type = idx.checked_type

        if structural_equal(x_type, pre_type):
            return idx
        return post


@RewriteDecorator
class ReorderDenseReshapeAdd(DFPatternCallback):
    """dense -> reshape -> add ===> dense -> add -> reshape"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        pattern = is_op("nn.dense")(wildcard(), is_constant())
        pattern = is_op("reshape")(pattern)
        self.pattern = is_op("add")(pattern, is_constant())

    def callback(self, pre, post, node_map):
        add = node_map[self.pattern][0]
        if isinstance(add.args[0], relay.Constant):
            add_const, reshape = add.args
        else:
            reshape, add_const = add.args
        data = add_const.data.numpy()
        if data.ndim > 1:
            return post
        dense = reshape.args[0]
        dense_add = dense + add_const
        return relay.reshape(dense_add, reshape.attrs.newshape)


@RewriteDecorator
class AdjustReduceKeepDim(DFPatternCallback):
    """Convert not keepdims reduce ops to keepdims ops"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        argmax = is_op("argmax")(wildcard())
        argmin = is_op("argmin")(wildcard())
        op_all = is_op("all")(wildcard())
        op_any = is_op("any")(wildcard())
        op_max = is_op("max")(wildcard())
        mean = is_op("mean")(wildcard())
        op_min = is_op("min")(wildcard())
        prod = is_op("prod")(wildcard())
        op_sum = is_op("sum")(wildcard())
        variance = is_op("variance")(wildcard(), wildcard())

        self.pattern = (
            argmax | argmin | op_all | op_any | op_max | mean | op_min | prod | op_sum | variance
        )

    def callback(self, pre, post, node_map):
        reduce = node_map[self.pattern][0]
        attrs = reduce.attrs
        if attrs.keepdims:
            return post

        new_attrs = {str(k): attrs[k] for k in attrs.keys()}
        new_attrs["keepdims"] = True
        new_attrs = ir.make_node(str(attrs).split("(")[0], **new_attrs)

        new_reduce = relay.Call(reduce.op, reduce.args, new_attrs, reduce.type_args, reduce.span)
        return relay.squeeze(new_reduce, new_attrs["axis"])


@RewriteDecorator
class AdjustMeanStd(DFPatternCallback):
    """Adjust Mean Std"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        variance = is_op("variance")(wildcard(), wildcard())
        squeeze = is_op("squeeze")(variance)
        sqrt = is_op("sqrt")(squeeze)

        self.pattern = sqrt

    def callback(self, pre, post, node_map):
        sqrt = node_map[self.pattern][0]
        squeeze = sqrt.args[0]
        variance = squeeze.args[0]

        new_sqrt = relay.sqrt(variance)
        return relay.squeeze(new_sqrt, squeeze.attrs.axis)


@RewriteDecorator
class Conv1DToConv2D(DFPatternCallback):
    """Convert Conv1D to Conv2D"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.pattern = is_op("nn.conv1d")(wildcard(), wildcard())

    def callback(self, pre, post, node_map):
        conv1d = node_map[self.pattern][0]
        attrs = conv1d.attrs
        data_layout = attrs.data_layout
        kernel_layout = attrs.kernel_layout
        conv2d_data_layout = "NCHW"
        conv2d_kernel_layout = "OHWI"
        data, kernel = conv1d.args

        squeeze_axis = 2
        if data_layout == "NCW":
            conv2d_data_layout = "NCHW"
            data = relay.expand_dims(data, 2)
            squeeze_axis = 2
        elif data_layout == "NWC":
            conv2d_data_layout = "NHWC"
            data = relay.expand_dims(data, 1)
            squeeze_axis = 1
        else:
            return post
        if kernel_layout == "OIW":
            conv2d_kernel_layout = "OIHW"
            kernel = relay.expand_dims(kernel, 2)
        elif kernel_layout == "OWI":
            conv2d_kernel_layout = "OHWI"
            kernel = relay.expand_dims(kernel, 1)
        else:
            return post

        strides = list(attrs.strides)
        padding = list(attrs.padding)
        dilation = list(attrs.dilation)

        kernel_size = list(attrs.kernel_size)

        strides = [1] + strides
        if len(padding) == 2:
            padding = [0, padding[0], 0, padding[1]]
        else:
            padding = [0, padding[0], 0, padding[0]]
        dilation = [1] + dilation
        kernel_size = [1] + kernel_size
        ret = relay.nn.conv2d(
            data,
            kernel,
            strides,
            padding,
            dilation,
            attrs.groups,
            attrs.channels,
            kernel_size,
            conv2d_data_layout,
            conv2d_kernel_layout,
        )
        return relay.squeeze(ret, [squeeze_axis])


@RewriteDecorator
class Expand1DOps(DFPatternCallback):
    """Convert add/sub/mul/div/exp inputs 1 dim to 2 dim"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        bin_op = is_op("divide") | is_op("multiply") | is_op("add") | is_op("subtract")
        unary_op = is_op("exp")
        self.pattern = bin_op(wildcard(), wildcard()) | unary_op(wildcard())

    def callback(self, pre, post, node_map):
        """ops has already broadcasted by other pass"""
        call = node_map[self.pattern][0]
        if call._checked_type_ is None:
            relay.transform.InferTypeLocal(call)
        dim = len(call.checked_type.shape)
        if dim > 1:
            return post
        new_args = []
        for arg in call.args:
            new_args.append(relay.expand_dims(arg, 0, 2 - dim))
        new_call = relay.Call(call.op, new_args, call.attrs)
        return relay.squeeze(new_call, list(range(2 - dim)))


@RewriteDecorator
class ClipPatternRewrite(DFPatternCallback):
    """Convert maximum(minimum) to clip"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        minimum = is_op("minimum")(wildcard(), is_constant())
        clip = is_op("maximum")(minimum, is_constant())
        self.pattern = clip

    def callback(self, pre, post, node_map):

        maximum = node_map[self.pattern][0]
        minimum, max_value = maximum.args
        param, min_value = minimum.args

        if _is_scalar(max_value) and _is_scalar(min_value):
            clip_min = float(max_value.data.numpy())
            clip_max = float(min_value.data.numpy())
            return relay.clip(param, clip_min, clip_max)

        return post


@RewriteDecorator
class RemoveZeroSizeRecipeForConcat(DFPatternCallback):
    """RemoveZeroSizeRecipeForConcat"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.pattern = is_op("concatenate")(wildcard())

    def callback(self, pre, post, node_map):
        concat = node_map[self.pattern][0]
        recipes = concat.args[0].fields

        def check_shape_zero(arg):
            shape = arg.checked_type.shape
            return any([val == 0 for val in shape])

        valid_recipes = [arg for arg in recipes if not check_shape_zero(arg)]
        if len(valid_recipes) == 1:
            return valid_recipes[0]
        elif len(valid_recipes) != len(recipes):
            relay.concatenate(valid_recipes, concat.attrs.axis)
        return post


@RewriteDecorator
class ExchangeSqueezeRequantizeAfterMean(DFPatternCallback):
    """mean + squeeze + requantize -> mean + requantize + squeeze"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        mean = is_op("mean")(wildcard())
        squeeze = is_op("squeeze")(mean)
        requantize = is_op("qnn.requantize")(
            squeeze, is_constant(), is_constant(), is_constant(), is_constant()
        )
        self.pattern = requantize

    def callback(self, pre, post, node_map):
        requantize = node_map[self.pattern][0]
        squeeze = requantize.args[0]
        requantize_args = requantize.args[1:]

        # Check whether overlapped for squeeze and requantize's axis.
        norm_axis = lambda ax, rank: ax + rank if ax < 0 else ax
        squeeze_input_shape = squeeze.args[0].checked_type.shape
        squeeze_axes = squeeze.attrs.axis
        squeeze_axes = [norm_axis(ax, len(squeeze_input_shape)) for ax in squeeze_axes]

        requantize_axis = requantize.attrs.axis
        requantize_axis = norm_axis(requantize_axis, len(squeeze_input_shape))
        if requantize_axis in squeeze_axes:
            return post

        new_in_data = relay.Call(
            requantize.op,
            [squeeze.args[0]] + requantize_args,
            requantize.attrs,
            requantize.type_args,
            requantize.span,
        )

        return relay.squeeze(new_in_data, squeeze.attrs.axis)


@RewriteDecorator
class RefineOpAttribute(DFPatternCallback):
    """RefineOpAttribute"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.pattern = is_op("tile")(wildcard())

    def callback(self, pre, post, node_map):
        op_call = node_map[self.pattern][0]
        ret = post
        if op_call.op == relay.op.get("tile"):
            if any([val.dtype != "int32" for val in op_call.attrs.reps]):
                reps = [int(val) for val in op_call.attrs.reps]
                ret = relay.tile(post.args[0], reps)
        return ret


@RewriteDecorator
class RewriteSearchSorted(DFPatternCallback):
    """Only suit for fastspeech2"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.pattern = is_op("searchsorted")(is_constant(), wildcard())

    def callback(self, pre, post, node_map):
        searchsorted = node_map[self.pattern][0]
        data, index_datas = searchsorted.args
        attrs = searchsorted.attrs
        right = attrs.right
        out_dtype = attrs.dtype
        if len(data.checked_type.shape) > 1:
            return post
        if len(index_datas.checked_type.shape) > 1:
            return post

        index_shape = index_datas.checked_type.shape
        # use argmax to rewrite searchsorted
        # argmax needs at least 2 dim
        const_data = data.data.numpy()
        data_dtype = const_data.dtype
        # make sure this value is big enough but not inf
        value_max = 99999.0
        value_max = max(np.max(np.abs(const_data)) * 100, value_max)
        value_max = np.array([value_max], dtype=data_dtype)
        new_data = np.concatenate([const_data, value_max], axis=0)

        # broadcast new_data to 2 dim
        new_data = np.expand_dims(new_data, -1)
        zeros_ = np.zeros([new_data.shape[0], int(index_shape[0])], dtype=data_dtype)
        new_data = new_data + zeros_
        new_data = relay.Constant(tvm.nd.array(new_data))

        if right:
            compare_data = index_datas < new_data
        else:
            compare_data = index_datas <= new_data

        out = relay.argmax(compare_data, axis=0, keepdims=True, select_last_index=False)
        out = relay.reshape(out, [index_shape[0]])
        relay.transform.InferTypeLocal(out)
        if out.checked_type.dtype != out_dtype:
            out = relay.cast(out, out_dtype)
            relay.transform.InferTypeLocal(out)
        return out


@RewriteDecorator
class RemoveRequantizeBeforeQnnConv2d(DFPatternCallback):
    """Simplify the requantize node before qnn.conv2d."""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        requantize = is_op("qnn.requantize")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        requantize_qnn_conv2d = is_op("qnn.conv2d")(
            requantize, is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        self.pattern = requantize_qnn_conv2d

    def callback(self, pre, post, node_map):
        qnn_conv2d = node_map[self.pattern][0]
        requantize = qnn_conv2d.args[0]
        in_dtype = tvm.DataType(requantize.args[0].checked_type.dtype)
        out_dtype = tvm.DataType(requantize.checked_type.dtype)
        in_scale = requantize.args[1].data.numpy()
        out_scale = requantize.args[3].data.numpy()
        if (
            str(in_dtype) not in ["int8", "uint8", "int16", "uint16"]
            or in_dtype.bits != out_dtype.bits
            or not np.allclose(in_scale, out_scale)
        ):
            return post
        new_args = [requantize.args[0], qnn_conv2d.args[1], requantize.args[2]]
        new_args += qnn_conv2d.args[3:]
        return relay.qnn.op.conv2d(*new_args, **qnn_conv2d.attrs)


@RewriteDecorator
class RemoveRequantizeAfterQuantize(DFPatternCallback):
    """Simplify the requantize node after quantize."""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        quantize = is_op("qnn.quantize")(wildcard(), is_constant(), is_constant())
        quantize_requantize = is_op("qnn.requantize")(
            quantize, is_constant(), is_constant(), is_constant(), is_constant()
        )
        self.pattern = quantize_requantize

    def callback(self, pre, post, node_map):
        requantize = node_map[self.pattern][0]
        in_scale = requantize.args[1].data.numpy()
        out_scale = requantize.args[3].data.numpy()
        if not np.allclose(in_scale, out_scale):
            return post
        out_dtype = requantize.attrs.out_dtype
        quantize = requantize.args[0]
        axis = quantize.attrs.axis
        return relay.qnn.op.quantize(
            quantize.args[0], quantize.args[1], requantize.args[4], axis, out_dtype
        )


@RewriteDecorator
class QnnConvertReduceMeanWithInconsistentIOScaleZp(DFPatternCallback):
    """Convert the reducemean with inconsistent input and output scale/zp.
    if mean == avgpool2d, cast+mean+qnn.requantize -> cast+avgpool2d+cast+qnn.add
    if mean != avgpool2d, cast+mean+qnn.requantize -> cast+mean+qnn.requantize+qnn.add
    """

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        cast = is_op("cast")(wildcard()).has_attr({"dtype": "int32"})
        mean = is_op("mean")(cast)
        requantize = is_op("qnn.requantize")(
            mean, is_constant(), is_constant(), is_constant(), is_constant()
        )
        self.pattern = requantize

    def callback(self, pre, post, node_map):
        requantize = node_map[self.pattern][0]
        mean = requantize.args[0]
        cast = mean.args[0]

        input_scale = requantize.args[1].data.numpy()
        input_zp = requantize.args[2].data.numpy()
        output_scale = requantize.args[3].data.numpy()
        output_zp = requantize.args[4].data.numpy()

        if np.allclose(input_scale, output_scale) and np.allclose(input_zp, output_zp):
            return post

        def get_converted_call():
            requant_with_same_scale_zp = relay.qnn.op.requantize(
                mean,
                requantize.args[1],
                requantize.args[2],
                requantize.args[1],
                requantize.args[2],
                **requantize.attrs,
            )

            in_shape = cast.checked_type.shape
            dims = len(in_shape)
            if dims != 4:
                return requant_with_same_scale_zp

            attrs = mean.attrs
            if not attrs.keepdims or not attrs.axis:
                return requant_with_same_scale_zp

            norm_axis = lambda ax, rank: ax + rank if ax < 0 else ax
            axis = [norm_axis(int(ax), dims) for ax in attrs.axis]
            if attrs.exclude:
                axis = [x for x in range(dims) if x not in axis]

            if axis == [1, 2]:
                layout_ = "NHWC"
            elif axis == [2, 3]:
                layout_ = "NCHW"
            else:
                return requant_with_same_scale_zp

            avg_pool2d = relay.nn.avg_pool2d(
                cast, (in_shape[axis[0]], in_shape[axis[1]]), layout=layout_, out_layout=layout_
            )

            return relay.op.cast(avg_pool2d, requantize.checked_type.dtype)

        # value of rhs of eltwise add
        const_node = np.zeros([int(i) for i in requantize.args[0].checked_type.shape])
        const_node = const_node.astype(cast.args[0].checked_type.dtype)
        add_value = relay.Constant(tvm.nd.array(const_node))
        # scale of rhs of eltwise add
        const_node = np.ones([int(i) for i in requantize.args[1].checked_type.shape])
        const_node = const_node.astype("float32")
        r_scale = relay.Constant(tvm.nd.array(const_node))
        # zp of rhs of eltwise add
        const_node = np.zeros([int(i) for i in requantize.args[2].checked_type.shape])
        const_node = const_node.astype("int32")
        r_zp = relay.Constant(tvm.nd.array(const_node))

        out_call = get_converted_call()
        return relay.qnn.op.add(
            out_call,
            add_value,
            requantize.args[1],
            requantize.args[2],
            r_scale,
            r_zp,
            requantize.args[3],
            requantize.args[4],
        )


@RewriteDecorator
class ReorderConv2dReshapeAddActivation(DFPatternCallback):
    """conv2d -> reshape -> add -> (relu) ===> conv2d -> add -> (relu) -> reshape"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
        reshape = is_op("squeeze")(conv2d)
        add = is_op("add")(reshape, is_constant())
        relu = is_op("nn.relu")(add)
        self.pattern = relu | add

    def callback(self, pre, post, node_map):
        last_op = node_map[self.pattern][0]
        if last_op.op.name == "nn.relu":
            add = last_op.args[0]
        else:
            add = last_op

        add_constant = add.args[0] if isinstance(add.args[0], relay.Constant) else add.args[1]
        size = add_constant.data.numpy().size
        if isinstance(add.args[0], relay.Constant):
            add_constant, squeeze = add.args
        else:
            squeeze, add_constant = add.args
        squeeze = add.args[1] if isinstance(add.args[0], relay.Constant) else add.args[0]
        conv2d = squeeze.args[0]
        data_layout = conv2d.attrs["data_layout"]
        if data_layout == "NHWC":
            add_constant = relay.reshape(add_constant, newshape=[size])
        elif data_layout == "NCHW":
            add_constant = relay.reshape(add_constant, newshape=[size, 1, 1])
        else:
            # if the layout is unknown, it can not reshape the bias data to a correct shape
            return post
        new_add = relay.add(conv2d, add_constant)
        if last_op.op.name == "nn.relu":
            new_relu = relay.nn.relu(new_add)
            new_squeeze = relay.squeeze(new_relu, axis=squeeze.attrs["axis"])
            return new_squeeze
        else:
            new_squeeze = relay.squeeze(new_add, axis=squeeze.attrs["axis"])
            return new_squeeze


@RewriteDecorator
class ChannelShuffleMerger(DFPatternCallback):
    """reshape + transpose + reshape ===> channel_shuffle"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        reshape0 = is_op("reshape")(wildcard())
        transpose = is_op("transpose")(reshape0)
        reshape1 = is_op("reshape")(transpose)
        self.pattern = reshape1

    def callback(self, pre, post, node_map):
        reshape1 = node_map[self.pattern][0]
        transpose = reshape1.args[0]
        reshape0 = transpose.args[0]

        in_shape = [int(x) for x in reshape0.args[0].checked_type.shape]
        out_shape = [int(x) for x in reshape0.checked_type.shape]
        out1_shape = [int(x) for x in reshape1.checked_type.shape]
        if len(in_shape) + 1 != len(out_shape) or in_shape != out1_shape:
            return post
        axes = [int(x) for x in transpose.attrs.axes]
        dim_num = len(axes)
        # NCHW
        if in_shape[1] == out_shape[1] * out_shape[2]:
            ref_axes = [0, 2, 1] + list(range(3, dim_num))
            group = min(out_shape[1], out_shape[2])
        # NHWC
        elif in_shape[-1] == out_shape[-1] * out_shape[-2]:
            ref_axes = list(range(0, dim_num - 2)) + [dim_num - 1, dim_num - 2]
            group = min(out_shape[-1], out_shape[-2])
        else:
            return post
        if axes != ref_axes:
            return post

        splits = 1
        return relay.op.contrib.aipu_compass.channel_shuffle(reshape0.args[0], group, splits)


@tvm.ir.transform.module_pass(opt_level=0)
class HintPatternRewrite:
    """
    Function to rewrite module by pattern
    """

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """try transform module"""

        update_mod = relay.transform.InferType()(mod)
        fold_constant = relay.transform.FoldConstant()
        infer_type = relay.transform.InferType()

        patterns = [
            Conv1DToConv2D(after_passes=fold_constant),
            ReorderConv2dReshapeAddActivation(),
            FoldDilatedConv2d(),
            RewriteSearchSorted(),
            ShapeOfTake(after_passes=fold_constant),
            DynamicFullConcatShape(after_passes=fold_constant),
            RecombineSoftmax(),
            EliminateIdentityOp(),
            Power2ToMul(after_passes=infer_type),
            ReorderDenseReshapeAdd(),
            AdjustReduceKeepDim(before_passes=infer_type),
            ExchangeSqueezeRequantizeAfterMean(before_passes=infer_type, after_passes=infer_type),
            RemoveZeroSizeRecipeForConcat(before_passes=infer_type),
            AdjustMeanStd(before_passes=infer_type),
            Expand1DOps(before_passes=infer_type, after_passes=infer_type),
            ClipPatternRewrite(),
            RefineOpAttribute(after_passes=infer_type),
            RemoveRequantizeBeforeQnnConv2d(before_passes=infer_type),
            RemoveRequantizeAfterQuantize(before_passes=infer_type),
            QnnConvertReduceMeanWithInconsistentIOScaleZp(before_passes=infer_type),
            ChannelShuffleMerger(),
        ]
        for pattern in patterns:
            update_mod = pattern(update_mod)

        return update_mod
