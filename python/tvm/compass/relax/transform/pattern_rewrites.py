# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
""" Rewrite module by pattern """
import numpy as np
from tvm import relax, ir
from tvm.compass.relax import op as compass_op
from tvm.tir import IntImm
from tvm.relax.dpl import is_op, wildcard, is_const, is_tuple_get_item
from .utils import get_inverse_axes


class MergeAdjacentReshape:
    """reshape + reshape --> reshape"""

    def __init__(self):
        self.reshape0 = is_op("relax.reshape")(wildcard(), wildcard())
        self.reshape1 = is_op("relax.reshape")(self.reshape0, wildcard())
        self.pattern = self.reshape1

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            inp = matches[self.reshape0].args[0]
            out_shape = matches[self.reshape1].struct_info.shape
            if all(isinstance(x, IntImm) for x in out_shape):
                return relax.op.reshape(inp, out_shape)
            return expr

        return self.pattern, rewriter


class MergeConvAddMulToConvAdd:
    """conv + add + mul --> conv + add
    src:        (WX + A) * M
    simplify:   (W * M) * X + (A * M)
    new_weight: (W * M), new_add: (A * M)

    This pattern rewrite only can be used in data layout: NHWC, kernel layout: OHWI.
    """

    def __init__(self):
        self.conv = is_op("relax.nn.conv2d")(wildcard(), is_const())
        self.add_const = is_const()
        self.add = is_op("relax.add")(self.conv, self.add_const)
        self.mul_const = is_const()
        self.mul = is_op("relax.multiply")(self.add, self.mul_const)
        self.pattern = self.mul

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            conv = matches[self.conv]
            weight_const = conv.args[1].data.numpy()
            add_const = matches[self.add_const].data.numpy()
            mul_const = matches[self.mul_const].data.numpy()

            if add_const.shape[-1] != weight_const.shape[0] or mul_const.size != 1:
                return expr

            new_weight_const = relax.const(weight_const * mul_const)
            new_add_const = relax.const(add_const * mul_const)

            new_conv = relax.Call(conv.op, [conv.args[0], new_weight_const], conv.attrs)
            new_add = relax.op.add(new_conv, new_add_const)
            return new_add

        return self.pattern, rewriter


class MergeAddSubToSub:
    """sub(add(x, const), const) ---> sub(x, const)"""

    def __init__(self):
        self.add = is_op("relax.add")(wildcard(), is_const())
        self.sub = is_op("relax.subtract")(self.add, is_const())
        self.pattern = self.sub

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            add, sub = matches[self.add], matches[self.sub]
            add_const = add.args[1].data.numpy()
            sub_const = sub.args[1].data.numpy()

            if add_const.shape != sub_const.shape:
                return expr

            new_const = relax.const(sub_const - add_const)
            new_sub = relax.Call(sub.op, [add.args[0], new_const], sub.attrs)
            return new_sub

        return self.pattern, rewriter


class MergeConstToConvWeight:
    """
    src:        [(WX - S) / D] * M + A
    simplify:   (WM / D) * X + [A - (SM / D)]
    new_weight: (WM / D), new_add: [A - (SM / D)]

    This pattern rewrite only can be used in data layout: NHWC, kernel layout: OHWI.
    """

    def __init__(self):
        self.conv = is_op("relax.nn.conv2d")(wildcard(), is_const())
        self.sub = is_op("relax.subtract")(self.conv, is_const())
        self.div = is_op("relax.divide")(self.sub, is_const())
        self.mul = is_op("relax.multiply")(self.div, is_const())
        self.add = is_op("relax.add")(self.mul, is_const())
        self.pattern = self.add

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            conv = matches[self.conv]
            weight = conv.args[1].data.numpy()
            sub_const = matches[self.sub].args[1].data.numpy().flatten()
            div_const = matches[self.div].args[1].data.numpy().flatten()
            mul_const = matches[self.mul].args[1].data.numpy().flatten()
            add_const = matches[self.add].args[1].data.numpy().flatten()

            if sub_const.shape != div_const.shape != mul_const.shape != add_const.shape:
                return expr
            elem_num = sub_const.shape[0]
            # Ensure elem_num == value_of_dim_'O', kernel layout: OHWI
            if elem_num != weight.shape[0]:
                return expr

            # new_weight_data = weight * mul_const / div_const
            # new_add_const_data = add_const - sub_const * mul_const / div_const
            # To align with the computation output in Relay, split the calculation
            # steps as follows:
            mid_out1 = 1 / div_const
            mid_out2 = mul_const * mid_out1
            mid_out3 = -sub_const
            mid_out4 = mid_out3 * mid_out2
            new_add_const_data = mid_out4 + add_const

            new_weight_data = weight * mid_out2.reshape([elem_num, 1, 1, 1])
            new_add_const_data = new_add_const_data.reshape([1, 1, 1, elem_num])
            new_weight = relax.const(new_weight_data, str(new_weight_data.dtype))
            new_add_const = relax.const(new_add_const_data, str(new_add_const_data.dtype))
            new_conv = relax.Call(conv.op, [conv.args[0], new_weight], conv.attrs)
            new_add = relax.op.add(new_conv, new_add_const)
            return new_add

        return self.pattern, rewriter


class MergeConstToFcWeight:
    """
    src:        [(WX + B - S) / D] * M + A
    simplify:   (WM / D) * X + [A - ((B - S) * M / D)]
    new_weight: (WM / D), new_add: [A - ((B - S) * M / D)]
    """

    def __init__(self):
        # matmul_bias_bn
        self.matmul = is_op("relax.matmul")(wildcard(), is_const())
        self.bias = is_op("relax.add")(self.matmul, is_const())
        self.sub = is_op("relax.subtract")(self.bias, is_const())
        self.div = is_op("relax.divide")(self.sub, is_const())
        self.mul = is_op("relax.multiply")(self.div, is_const())
        self.matmul_bias_bn = is_op("relax.add")(self.mul, is_const())
        self.pattern = self.matmul_bias_bn

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            matmul = matches[self.matmul]
            weight = matmul.args[1].data.numpy()
            sub_const = matches[self.sub].args[1].data.numpy().flatten()
            div_const = matches[self.div].args[1].data.numpy().flatten()
            mul_const = matches[self.mul].args[1].data.numpy().flatten()
            add_const = matches[self.matmul_bias_bn].args[1].data.numpy().flatten()
            bias_const = matches[self.matmul_bias_bn].args[1].data.numpy().flatten()

            if (
                sub_const.shape
                != bias_const.shape
                != div_const.shape
                != mul_const.shape
                != add_const.shape
            ):
                return expr

            elem_num = sub_const.shape[0]
            if elem_num != weight.shape[1]:
                return expr

            # new_weight_data = weight * mul_const / div_const
            # new_add_const_data = add_const + (bias_const - sub_const) * mul_const / div_const
            # To align with the computation output in Relay, split the calculation
            # steps as follows:
            mid_out1 = 1 / div_const
            mid_out2 = mul_const * mid_out1
            mid_out3 = -sub_const if bias_const is None else bias_const - sub_const
            mid_out4 = mid_out3 * mid_out2
            new_add_const_data = mid_out4 + add_const

            new_weight_data = weight * mid_out2
            new_weight = relax.const(new_weight_data, str(new_weight_data.dtype))
            new_add_const = relax.const(new_add_const_data, str(new_add_const_data.dtype))
            new_matmul = relax.Call(matmul.op, [matmul.args[0], new_weight], matmul.attrs)
            new_add = relax.op.add(new_matmul, new_add_const)
            return new_add

        return self.pattern, rewriter


class MergeRehapeTransReshape:
    """ChannelShuffle"""

    def __init__(self):
        self.reshape0 = is_op("relax.reshape")(wildcard(), wildcard())
        self.permute = is_op("relax.permute_dims")(self.reshape0)
        self.reshape1 = is_op("relax.reshape")(self.permute, wildcard())
        self.pattern = self.reshape1

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            reshape0 = matches[self.reshape0]
            permute = matches[self.permute]
            reshape1 = matches[self.reshape1]

            in_shape = [int(x) for x in reshape0.args[0].struct_info.shape]
            out_shape = [int(x) for x in reshape0.struct_info.shape]
            out1_shape = [int(x) for x in reshape1.struct_info.shape]
            if len(in_shape) + 1 != len(out_shape) or in_shape != out1_shape:
                return expr
            axes = [int(x) for x in permute.attrs.axes]
            dim_num = len(axes)

            # Guess input layout from input shape and output shape.
            axis = None
            if in_shape[1] == out_shape[1] * out_shape[2]:
                # input layout: NCHW
                ref_axes = [0, 2, 1] + list(range(3, dim_num))
                group = min(out_shape[1], out_shape[2])
                axis = 1
            elif in_shape[-1] == out_shape[-1] * out_shape[-2]:
                # input layout: NHWC
                ref_axes = list(range(0, dim_num - 2)) + [dim_num - 1, dim_num - 2]
                group = min(out_shape[-1], out_shape[-2])
                axis = dim_num - 1
            else:
                return expr

            if axes != ref_axes:
                return expr

            return compass_op.channel_shuffle(reshape0.args[0], group, axis, splits=1)

        return self.pattern, rewriter


class Conv1DToConv2D:
    """conv1d ===> expand_dims -> conv2d -> squeeze."""

    def __init__(self):
        self.conv1d = is_op("relax.nn.conv1d")(wildcard(), is_const())
        self.pattern = self.conv1d

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            conv1d = matches[self.conv1d]
            attrs = conv1d.attrs

            if (
                attrs.kernel_layout != "OIW"
                or attrs.data_layout != "NCW"
                or attrs.out_layout != "NCW"
            ):
                return expr

            weight = conv1d.args[1].data.numpy()
            weight_shape = list(weight.shape)
            new_weight = relax.const(weight.reshape(*weight_shape[:2], 1, *weight_shape[2:]))

            new_inp0 = relax.op.expand_dims(conv1d.args[0], 2)
            conv2d = relax.op.nn.conv2d(
                new_inp0,
                new_weight,
                [1] + attrs.strides[:],
                [0, 0] + conv1d.attrs.padding[:],  # top, bottom, left, right
                [1] + attrs.dilation[:],
                attrs.groups,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="NCHW",
                out_dtype=attrs.out_dtype,
            )
            return relax.op.squeeze(conv2d, axis=2)

        return self.pattern, rewriter


class MergeMultiplyToConvWeight:
    """
    src:        WX * M
    simplify:   (W * M) * X
    new_weight: (W * M)

    This pattern rewrite only can be used in data layout: NHWC, kernel layout: OHWI.
    """

    def __init__(self):
        self.conv = is_op("relax.nn.conv2d")(wildcard(), is_const())
        self.mul = is_op("relax.multiply")(self.conv, is_const())
        self.pattern = self.mul

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            conv = matches[self.conv]
            weight = conv.args[1].data.numpy()
            mul_const = matches[self.mul].args[1].data.numpy().reshape(-1, 1, 1, 1)

            # Weight layout: OHWI, ensure const_num == 'O'_num
            if mul_const.shape[0] != weight.shape[0]:
                return expr

            # data layout: NHWC
            new_weight_data = weight * mul_const
            new_weight = relax.const(new_weight_data, str(new_weight_data.dtype))
            new_conv = relax.Call(conv.op, [conv.args[0], new_weight], conv.attrs)

            return new_conv

        return self.pattern, rewriter


class EliminateUselessPermuteDims:
    """permute_dims(x, [0,1,2,3]) --> x"""

    def __init__(self):
        self.permute_dims = is_op("relax.permute_dims")(wildcard())
        self.pattern = self.permute_dims

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            permute_dims = matches[self.permute_dims]
            axes = [int(x) for x in permute_dims.attrs.axes]
            if any(axes[i] != i for i in range(permute_dims.args[0].struct_info.ndim)):
                return expr
            return permute_dims.args[0]

        return self.pattern, rewriter


class MergeQuantCast:
    """astype(quant(inp)) --> quant(inp)"""

    def __init__(self):
        self.quant = is_op("relax.quantize")(wildcard(), is_const(), is_const())
        self.astype = is_op("relax.astype")(self.quant)
        self.pattern = self.astype

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):  # pylint: disable=unused-argument
            quant = matches[self.quant]
            astype = matches[self.astype]
            if quant.struct_info.dtype == astype.struct_info.dtype:
                return quant
            return relax.op.qdq.quantize(*quant.args, quant.attrs.axis, astype.attrs.dtype)

        return self.pattern, rewriter


class MergePermMean:
    """mean(permute_dims(inp)) --> mean(inp)"""

    def __init__(self):
        self.perm = is_op("relax.permute_dims")(wildcard())
        self.mean = is_op("relax.mean")(self.perm)
        self.pattern = self.mean

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):  # pylint: disable=unused-argument
            perm = matches[self.perm]
            mean = matches[self.mean]
            axes = [int(x) for x in perm.attrs.axes]
            new_axis = [axes[int(x)] for x in mean.attrs.axis]
            new_mean = relax.op.mean(perm.args[0], new_axis, mean.attrs.keepdims)
            return new_mean

        return self.pattern, rewriter


class MergeExplicitPad:
    """candidate_op(pad(inp)) --> candidate_op(inp)
    candidate_op:avg_pool, max_pool, conv2d, conv3d, dequantize+conv2d
    Todo: max_pool, conv3d, reference to fold_explicit_padding.cc

    This pattern rewrite only can be used in data layout: NHWC, kernel layout: OHWI.
    """

    def __init__(self):
        self.pad = is_op("relax.nn.pad")(wildcard())
        self.permute_dims = is_op("relax.permute_dims")(self.pad)
        self.dequantize = is_op("relax.dequantize")(self.pad, is_const(), is_const())
        self.pad_or_tpad_or_dpad = self.pad | self.permute_dims | self.dequantize
        self.avgpool = is_op("relax.nn.avg_pool2d")(self.pad_or_tpad_or_dpad)
        self.conv2d = is_op("relax.nn.conv2d")(self.pad_or_tpad_or_dpad, wildcard())
        self.pattern = self.avgpool | self.conv2d

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        # pylint: disable=inconsistent-return-statements
        def rewriter(expr, matches):  # pylint: disable=unused-argument
            pad = matches[self.pad]
            attrs = pad.attrs
            pad_width = pad.attrs.pad_width
            if len(pad_width) != 8 or attrs.pad_mode != "constant":
                return expr

            dequantize = matches.get(self.dequantize)
            pv_golden = 0.0
            if dequantize:
                pv_golden = dequantize.args[2].data.numpy()

            if attrs.pad_value != pv_golden:
                return expr

            permute_dims = matches.get(self.permute_dims)
            if permute_dims and list(permute_dims.attrs.axes) != [0, 2, 3, 1]:
                return expr
            # pad width layout: NCHW if permute_dims else NHWC, we need pads in HW.
            pads = pad_width[4:] if permute_dims else pad_width[2:6]

            avgpool = matches.get(self.avgpool)
            conv2d = matches.get(self.conv2d)
            if permute_dims:
                inp = relax.op.permute_dims(pad.args[0], permute_dims.attrs.axes)
            elif dequantize:
                axis = dequantize.attrs.axis
                out_dtype = dequantize.attrs.out_dtype
                new_args = [dequantize.args[1], dequantize.args[2], axis, out_dtype]
                inp = relax.op.qdq.dequantize(pad.args[0], *new_args)
            else:
                inp = pad.args[0]

            if avgpool:
                attrs = avgpool.attrs

                return relax.op.nn.avg_pool2d(
                    inp,
                    attrs.pool_size,
                    attrs.strides,
                    pads,  # top, left, bottom, right
                    attrs.dilation,
                    attrs.ceil_mode,
                    attrs.count_include_pad,
                    attrs.layout,
                    attrs.out_layout,
                )

            if conv2d:
                attrs = conv2d.attrs
                new_padding = [pads[0], pads[2], pads[1], pads[3]]
                return relax.op.nn.conv2d(
                    inp,
                    conv2d.args[1],
                    attrs.strides,
                    new_padding,  # top, bottom, left, right
                    attrs.dilation,
                    attrs.groups,
                    attrs.data_layout,
                    attrs.kernel_layout,
                    attrs.out_layout,
                    attrs.out_dtype,
                )

        return self.pattern, rewriter


class UpdateMatmul:
    """Update matmul inputs rank."""

    def __init__(self):
        self.matmul = is_op("relax.matmul")(wildcard(), wildcard())
        self.pattern = self.matmul

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            matmul = matches[self.matmul]
            inp0, inp1 = matmul.args
            a_rank = len(inp0.struct_info.shape)
            b_rank = len(inp1.struct_info.shape)
            if a_rank > 2 and b_rank == 2:
                new_a = relax.op.reshape(inp0, [-1, inp0.struct_info.shape[-1]])
                out = relax.op.matmul(new_a, inp1)
                return relax.op.reshape(out, expr.struct_info.shape)
            return expr

        return self.pattern, rewriter


class ReorderMatmulReshapeAdd:
    """add(reshape(matmul(inp))) --> reshape(add(matmul((inp)))"""

    def __init__(self):
        self.matmul = is_op("relax.matmul")(wildcard(), is_const())
        self.reshape = is_op("relax.reshape")(self.matmul, wildcard())
        self.add_const = is_const()
        self.add = is_op("relax.add")(self.reshape, self.add_const)
        self.pattern = self.add

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            matmul = matches[self.matmul]
            reshape = matches[self.reshape]
            add_const = matches[self.add_const]
            data = add_const.data.numpy()
            if data.ndim > 1:
                return expr
            matmul_add = matmul + add_const
            return relax.op.reshape(matmul_add, reshape.args[1])

        return self.pattern, rewriter


class ReorderConv2dReshapeAddActivation:
    """conv2d -> reshape -> add -> (relu) ===> conv2d -> add -> (relu) -> reshape"""

    def __init__(self):
        self.conv2d = is_op("relax.nn.conv2d")(wildcard(), is_const())
        self.reshape = is_op("relax.squeeze")(self.conv2d)
        self.add = is_op("relax.add")(self.reshape, is_const())
        self.relu = is_op("relax.nn.relu")(self.add)
        self.pattern = self.relu | self.add

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            conv2d = matches[self.conv2d]
            add_const = matches[self.add].args[1].data.numpy()
            if conv2d.attrs.data_layout != "NCHW":
                return expr

            add_const_expr = relax.const(add_const.reshape([add_const.size, 1, 1]))
            ret = relax.op.add(conv2d, add_const_expr)
            if matches.get(self.relu):
                ret = relax.op.nn.relu(ret)
            return relax.op.squeeze(ret, axis=matches[self.reshape].attrs.axis)

        return self.pattern, rewriter


class ReorderBinaryOpsConstArgs:
    """bin_op(const, non_const) ===> bin_op(non_const, const)
    When some binary op (excluding order-sensitive operations such as divide and subtract) are
    given in the order (constant, non-constant), swap their positions so that the non-constant
    operand comes first (on the left-hand side) and the constant operand comes second
    (on the right-hand side).
    """

    def __init__(self):
        self.add = is_op("relax.add")(is_const(), wildcard())
        self.mul = is_op("relax.multiply")(is_const(), wildcard())
        self.max = is_op("relax.maximum")(is_const(), wildcard())
        self.min = is_op("relax.minimum")(is_const(), wildcard())
        # pylint: disable=unsupported-binary-operation
        self.pattern = self.add | self.mul | self.max | self.min

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            bin_call = matches[self.pattern]
            args0, args1 = bin_call.args
            if isinstance(args0, relax.Constant) and not isinstance(args1, relax.Constant):
                return relax.Call(bin_call.op, [args1, args0], bin_call.attrs)
            return expr

        return self.pattern, rewriter


class EliminateIdentityOp:
    """Eliminates expressions that are equivalent to identity."""

    def __init__(self):
        self.idx = wildcard()
        self.reshape = is_op("relax.reshape")(self.idx, wildcard())
        self.permute_dims = is_op("relax.permute_dims")(self.idx)
        split = is_op("relax.split")(self.idx)
        self.split = is_tuple_get_item(split, 0)
        self.resize2d = is_op("relax.image.resize2d")(self.idx, wildcard())
        self.concat = is_op("relax.concat")(self.idx)
        self.avg_pool2d = is_op("relax.nn.avg_pool2d")(self.idx)
        self.broadcast_to = is_op("relax.broadcast_to")(self.idx, wildcard())

        # pylint: disable=unsupported-binary-operation
        self.pattern = (
            self.reshape
            | self.permute_dims
            | self.split
            | self.resize2d
            | self.concat
            | self.avg_pool2d
            | self.broadcast_to
        )

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            idx = matches[self.idx]
            permute_dims = matches.get(self.permute_dims)
            concat = matches.get(self.concat)
            avg_pool2d = matches.get(self.avg_pool2d)

            if permute_dims:
                if permute_dims.attrs.axes is None:
                    return expr
                axes = [int(axis) for axis in permute_dims.attrs.axes]
                if axes != list(range(len(axes))):
                    return expr

            if concat:
                return idx[0] if len(idx) == 1 else expr

            if avg_pool2d:
                pool_size_set = np.unique(avg_pool2d.attrs.pool_size)
                is_pool1x1 = pool_size_set.size == 1 and pool_size_set == np.array([1])
                padding_set = np.unique(avg_pool2d.attrs.padding)
                is_pad_0 = padding_set.size == 1 and padding_set == np.array([0])
                stride_set = np.unique(avg_pool2d.attrs.strides)
                is_stride_1 = stride_set.size == 1 and stride_set == np.array([1])
                if not (is_pool1x1 and is_pad_0 and is_stride_1):
                    return expr

            pre_type = expr.struct_info
            x_type = idx.struct_info
            if ir.structural_equal(pre_type, x_type):
                return idx
            return expr

        return self.pattern, rewriter


class SimplifyAddZeroMulOne:
    """add(inp, 0) / mul(inp, 1) ===> inp"""

    def __init__(self):
        self.inp = wildcard()
        self.add_const = is_const()
        self.add_0 = is_op("relax.add")(self.inp, self.add_const)
        self.mul_const = is_const()
        self.mul_1 = is_op("relax.multiply")(self.inp, self.mul_const)
        self.pattern = self.add_0 | self.mul_1

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            inp = matches[self.inp]
            if matches.get(self.add_0):
                add_const = matches[self.add_const]
                np_const = add_const.data.numpy()
                if np.allclose(np_const, 0):
                    return inp
            elif matches.get(self.mul_1):
                mul_const = matches[self.mul_const]
                np_const = mul_const.data.numpy()
                if np.allclose(np_const, 1):
                    return inp
            return expr

        return self.pattern, rewriter


class ConvertToReshape:
    """Convert some ops to reshape."""

    def __init__(self):
        self.inp = wildcard()
        self.squeeze = is_op("relax.squeeze")(self.inp)
        self.expand_dims = is_op("relax.expand_dims")(self.inp)
        self.permute_dims = is_op("relax.permute_dims")(self.inp)
        self.pattern = self.squeeze | self.expand_dims | self.permute_dims

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            inp = matches[self.inp]
            out_shape = [int(x) for x in expr.struct_info.shape]
            if matches.get(self.permute_dims):
                for dim in out_shape:
                    if dim != 1 and dim != np.prod(out_shape):
                        return expr

            return relax.op.reshape(inp, out_shape)

        return self.pattern, rewriter


class SimplifyTransReshapeTrans:
    """Simplify permute_dims -> reshape -> permute_dims."""

    def __init__(self):
        self.inp = wildcard()
        self.perm0 = is_op("relax.permute_dims")(self.inp)
        self.reshape = is_op("relax.reshape")(self.perm0, wildcard())
        self.perm1 = is_op("relax.permute_dims")(self.reshape)
        self.pattern = self.perm1

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            perm0 = matches[self.perm0]
            inp_shape = [int(x) for x in perm0.args[0].struct_info.shape]
            out_shape = [int(x) for x in expr.struct_info.shape]

            if len(inp_shape) == 4 and len(out_shape) == 3:
                if inp_shape == [out_shape[0], 1, *out_shape[1:]]:
                    return relax.op.reshape(matches[self.inp], out_shape)

            if len(inp_shape) == 3 and len(out_shape) == 4:
                if out_shape == [inp_shape[0], 1, *inp_shape[1:]]:
                    return relax.op.reshape(perm0, out_shape)

            return expr

        return self.pattern, rewriter


class RevertReshape:
    """Convert some_op(reshape) -> reshape(some_op)."""

    def __init__(self):
        self.inp = wildcard()
        self.reshape = is_op("relax.reshape")(self.inp, wildcard())
        self.log_softmax = is_op("relax.nn.log_softmax")(self.reshape)
        self.reduce = is_op("relax.max")(self.reshape)
        self.pattern = self.log_softmax | self.reduce

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            inp = matches[self.inp]
            inp_shape = [int(x) for x in matches[self.reshape].args[0].struct_info.shape]
            reshape_shape = [int(x) for x in matches[self.reshape].struct_info.shape]
            out_shape = [int(x) for x in expr.struct_info.shape]

            if len(inp_shape) == 4 and len(out_shape) == 2:
                if matches.get(self.log_softmax):
                    axis = matches[self.log_softmax].attrs.axis
                    if (
                        len(reshape_shape) == 2
                        and reshape_shape[axis] == inp_shape[-1]
                        and np.prod(inp_shape[:-1]) == np.prod(reshape_shape[:axis])
                    ):
                        log_softmax = relax.op.nn.log_softmax(inp, axis=len(inp_shape) - 1)
                        return relax.op.reshape(log_softmax, out_shape)

            if len(inp_shape) == 4 and len(out_shape) == 3:
                if matches.get(self.reduce):
                    attrs = matches[self.reduce].attrs
                    axis = attrs.axis
                    if len(axis) != 1:
                        return expr

                    axis = int(axis[0])
                    if (
                        len(reshape_shape) == 3
                        and reshape_shape[axis] == inp_shape[-1]
                        and np.prod(inp_shape[:-1]) == np.prod(reshape_shape[:axis])
                    ):
                        new_reduce = relax.op.max(inp, len(inp_shape) - 1, keepdims=attrs.keepdims)
                        return relax.op.reshape(new_reduce, out_shape)
            return expr

        return self.pattern, rewriter


class AddToMul:
    """Add(x, x) --> Mul(x, 2)"""

    def __init__(self):
        self.inp = wildcard()
        self.add = is_op("relax.add")(self.inp, self.inp)
        self.pattern = self.add

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):  # pylint: disable=unused-argument
            inp = matches[self.inp]
            return relax.op.multiply(inp, relax.const(2, inp.struct_info.dtype))

        return self.pattern, rewriter


class BroadcastToTile:
    """broadcast_to --> tile"""

    def __init__(self):
        self.bcast = is_op("relax.broadcast_to")(wildcard(), wildcard())
        self.pattern = self.bcast

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):  # pylint: disable=unused-argument
            bcast = matches[self.bcast]
            inp = bcast.args[0]

            input_shape = [int(x) for x in inp.struct_info.shape.values]
            if all(isinstance(x, IntImm) for x in bcast.args[1]):
                to_shape = [int(x) for x in bcast.args[1]]
                if len(to_shape) > len(input_shape):
                    delta = len(to_shape) - len(input_shape)
                    input_shape = [1] * delta + input_shape
                    inp = relax.op.reshape(inp, input_shape)

                reps = [to // old for to, old in zip(to_shape, input_shape)]
                if reps[0] == 1 and len(set(reps)) == 1:
                    return inp
                return relax.op.tile(inp, reps)
            return expr

        return self.pattern, rewriter


class FoldDilatedConv2d:
    """space_to_batch_nd + conv2d + batch_to_space_nd --> conv2d"""

    def __init__(self):
        self.inp = wildcard()
        self.s2b = is_op("relax.nn.space_to_batch_nd")(self.inp)
        self.conv = is_op("relax.nn.conv2d")(self.s2b, is_const())
        self.b2s = is_op("relax.nn.batch_to_space_nd")(self.conv)
        self.pattern = self.b2s

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):  # pylint: disable=unused-argument
            inp = matches[self.inp]
            s2b = matches[self.s2b]
            conv = matches[self.conv]
            b2s = matches[self.b2s]

            s2b_block = s2b.attrs.block_shape
            b2s_block = b2s.attrs.block_shape
            src_dilation = conv.attrs.dilation
            src_padding = conv.attrs.padding
            if (
                not all(x == y for x, y in zip(s2b_block, b2s_block))
                or len(s2b_block) != 2
                or not all(x == 1 for x in src_dilation)
                or not all(x == 0 for x in src_padding)
            ):
                return expr

            s2b_pad = s2b.attrs.paddings
            b2s_crop = b2s.attrs.crops
            if len(s2b_pad) != 2 or len(b2s_crop) != 2:
                return expr

            pad_h, pad_w = s2b_pad
            crop_h, crop_w = b2s_crop
            pad_top = pad_h[0] - crop_h[0]
            pad_bottom = pad_h[1] - crop_h[1]
            pad_left = pad_w[0] - crop_w[0]
            pad_right = pad_w[1] - crop_w[1]

            new_dilation = [s2b_block[0], s2b_block[1]]
            new_padding = [pad_top, pad_left, pad_bottom, pad_right]
            return relax.op.nn.conv2d(
                inp,
                conv.args[1],
                conv.attrs.strides,
                new_padding,
                new_dilation,
                conv.attrs.groups,
                conv.attrs.data_layout,
                conv.attrs.kernel_layout,
                conv.attrs.out_layout,
                conv.attrs.out_dtype,
            )

        return self.pattern, rewriter


class BindDequantQuant:
    """quantize(dequantize(inp)) --> requantize(x)"""

    def __init__(self):
        self.inp = wildcard()
        self.dequantize = is_op("relax.dequantize")(self.inp, is_const(), is_const())
        self.quantize = is_op("relax.quantize")(self.dequantize, is_const(), is_const())
        self.pattern = self.quantize

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):  # pylint: disable=unused-argument
            inp = matches[self.inp]
            dequantize = matches[self.dequantize]
            quantize = matches[self.quantize]
            attrs = quantize.attrs
            if dequantize.attrs.out_dtype != "float32":
                return expr
            if attrs.axis != dequantize.attrs.axis:
                return expr

            inp_scale, inp_zp = dequantize.args[1:]
            out_scale, out_zp = quantize.args[1:]
            # Some qnn op like: q + resize + dq, can eliminate resize by EliminateIdentityOp,
            # so eliminate q + dq here.
            if (
                inp.struct_info.dtype == expr.struct_info.dtype
                and np.allclose(inp_scale.data.numpy(), out_scale.data.numpy())
                and np.allclose(inp_zp.data.numpy(), out_zp.data.numpy())
            ):
                return inp
            args = [inp, inp_scale, inp_zp, out_scale, out_zp, attrs.axis, attrs.out_dtype]
            return compass_op.requantize(*args)

        return self.pattern, rewriter


class UnBindDequantQuant:
    """requantize(x) --> quantize(dequantize(inp))"""

    def __init__(self):
        self.inp = wildcard()
        self.requantize = is_op("relax.requantize")(
            self.inp, is_const(), is_const(), is_const(), is_const()
        )
        self.pattern = self.requantize

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):  # pylint: disable=unused-argument
            inp = matches[self.inp]
            req = matches[self.requantize]
            axis = req.attrs.axis
            out_dtype = req.attrs.out_dtype
            deq = relax.op.qdq.dequantize(inp, req.args[1], req.args[2], axis)
            return relax.op.qdq.quantize(deq, req.args[3], req.args[4], axis, out_dtype)

        return self.pattern, rewriter


class SimplifyConsecutivePermuteDims:
    """permute_dims1(permute_dims0(inp)) --> inp"""

    def __init__(self):
        self.permute_dims0 = is_op("relax.permute_dims")(wildcard())
        self.permute_dims1 = is_op("relax.permute_dims")(self.permute_dims0)
        self.pattern = self.permute_dims1

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            permute_dims0 = matches[self.permute_dims0]
            permute_dims1 = matches[self.permute_dims1]
            axes0 = permute_dims0.attrs.axes
            axes1 = permute_dims1.attrs.axes
            if all(x == y for x, y in zip(axes0, get_inverse_axes(axes1))):
                return permute_dims0.args[0]
            return expr

        return self.pattern, rewriter


class AdjustQnnMeanKeepDim:
    """qnn_mean(inp) without keepdim --> qnn_mean(inp) with keepdim + reshape"""

    def __init__(self):
        self.inp = wildcard()
        self.deq = is_op("relax.dequantize")(self.inp, is_const(), is_const())
        self.mean = is_op("relax.mean")(self.deq)
        self.quant = is_op("relax.quantize")(self.mean, is_const(), is_const())
        self.pattern = self.quant

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            mean = matches[self.mean]
            if mean.attrs.keepdims:
                return expr
            deq = matches[self.deq]
            quant = matches[self.quant]
            new_mean = relax.op.mean(deq, mean.attrs.axis, True)
            new_q = relax.Call(quant.op, [new_mean, *quant.args[1:]], quant.attrs)
            reshape = relax.op.reshape(new_q, mean.struct_info.shape)
            return reshape

        return self.pattern, rewriter


class ConvertMeanToPool:
    """qnn_mean(inp) --> qnn_add(avg_pool2d(inp), 0)"""

    def __init__(self):
        self.inp = wildcard()
        self.deq = is_op("relax.dequantize")(self.inp, is_const(), is_const())
        self.mean = is_op("relax.mean")(self.deq)
        self.quant = is_op("relax.quantize")(self.mean, is_const(), is_const())
        self.pattern = self.quant

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            mean = matches[self.mean]
            attrs = mean.attrs
            if not attrs.keepdims or not attrs.axis:
                return expr
            in_shape = [int(x) for x in mean.args[0].struct_info.shape]
            dims = len(in_shape)
            norm_axis = lambda ax, rank: ax + rank if ax < 0 else ax
            axis = [norm_axis(int(ax), dims) for ax in attrs.axis]
            if dims != 4 or axis != [1, 2]:
                return expr
            kernel_x = in_shape[2]
            kernel_y = in_shape[1]
            max_int32 = 2**31 - 1
            input_data_size = 32
            if (
                (kernel_x > 65 or kernel_y > 65)
                and kernel_x * kernel_y >= 28 * 1024
                and kernel_x * kernel_y * input_data_size >= max_int32
            ):
                return expr

            inp = matches[self.inp]
            deq = matches[self.deq]
            quant = matches[self.quant]
            avgp = relax.op.nn.avg_pool2d(inp, (in_shape[1], in_shape[2]), layout="NHWC")
            new_dq = relax.Call(deq.op, [avgp, *deq.args[1:]], deq.attrs)
            mean_out_shape = [int(x) for x in mean.struct_info.shape]
            const_data = np.zeros(mean_out_shape, inp.struct_info.dtype)
            add_const = relax.const(const_data, inp.struct_info.dtype)
            const_scale = relax.const(1.0, "float32")
            const_zp = relax.const(0, "int32")
            dq_const = relax.op.qdq.dequantize(add_const, const_scale, const_zp)
            add = relax.op.add(new_dq, dq_const)
            new_q = relax.Call(quant.op, [add, *quant.args[1:]], quant.attrs)
            return new_q

        return self.pattern, rewriter


class AdjustArgMinMaxKeepDim:
    """Covnert not keepdims argmin/argmax to keepdims argmin/argmax"""

    def __init__(self):
        self.argmin = is_op("relax.argmin")(wildcard())
        self.argmax = is_op("relax.argmax")(wildcard())
        self.pattern = self.argmin | self.argmax

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            call = matches[self.pattern]
            inp = call.args[0]
            attrs = call.attrs
            if attrs.keepdims:
                return expr

            new_attrs = {str(k): attrs[k] for k in attrs.keys()}
            new_attrs["keepdims"] = True
            if attrs.axis < 0:
                new_attrs["axis"] = len(inp.struct_info.shape.values) + attrs.axis
            new_attrs = ir.make_node(str(attrs).split("(")[0], **new_attrs)

            new_call = relax.Call(call.op, [inp], new_attrs)
            return relax.op.squeeze(new_call, new_attrs.axis)

        return self.pattern, rewriter
