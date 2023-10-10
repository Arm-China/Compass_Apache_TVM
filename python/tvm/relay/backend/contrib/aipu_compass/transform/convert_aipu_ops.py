# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
"""Convert some ops to other equivalents."""
import operator
import functools
import numpy as np
import tvm
from tvm import relay
from tvm.aipu.logger import WARN


def _prod(arr):
    return functools.reduce(operator.mul, arr, 1)


def _infer_shape(expr):
    return relay.transform.InferType()(tvm.IRModule.from_expr(expr))["main"].body.checked_type.shape


class AIPUOpsConvertor(relay.ExprMutator):
    """Convert some ops to other equivalents."""

    def _is_dynamic(self, call):
        datatype = call._checked_type_
        if datatype is None:
            mod = tvm.IRModule.from_expr(call)
            mod = relay.transform.InferType()(mod)
            datatype = mod["main"].body.checked_type
        return relay.ty.is_dynamic(datatype)

    def _broadcast_inputs(self, call):
        shapes = [type_arg.shape for type_arg in call.type_args[:2]]
        try:
            shapes = [[int(idx) for idx in shape] for shape in shapes]
        except TypeError:
            return None
        max_dim = max([len(shape) for shape in shapes])
        origin_args = list(call.args[:2])
        args = []
        new_shapes = []
        for idx, shape in enumerate(shapes):
            arg = origin_args[idx]
            new_shape = shape
            if len(shape) != max_dim:
                new_shape = [1] * (max_dim - len(shape)) + shape
                arg = relay.reshape(arg, new_shape)
            args.append(arg)
            new_shapes.append(new_shape)

        new_shape = []
        for dim in range(max_dim):
            cur_dim_shape = [shape[dim] for shape in new_shapes]
            dim_max_size = max(cur_dim_shape)
            new_shape.append(dim_max_size)

        new_args = []
        for arg, shape in zip(args, new_shapes):
            newarg = arg
            if shape != new_shape:
                reps = [dim0 // dim1 for dim0, dim1 in zip(new_shape, shape)]
                newarg = relay.tile(arg, reps)
            new_args.append(newarg)
        if new_args == origin_args:
            return call
        new_args += call.args[2:]
        return relay.Call(call.op, new_args, call.attrs)

    def visit_call(self, call):

        ret = super().visit_call(call)

        if isinstance(call.op, relay.Function):
            attrs = call.op.attrs
            if "Composite" in attrs and "aipu_compass" in attrs["Composite"]:
                new_args = ret.args
                return relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)

        if self._is_dynamic(ret):
            return ret

        _convert_to_reshape_op = [
            relay.op.get("squeeze"),
            relay.op.get("nn.batch_flatten"),
            relay.op.get("expand_dims"),
            relay.op.get("transpose"),
        ]

        if ret.op in _convert_to_reshape_op:
            if ret.op == relay.op.get("transpose"):
                out_shapes = np.array(call.checked_type.shape)
                for dim in out_shapes:
                    if dim != 1 and dim != np.prod(out_shapes):
                        return ret
            return relay.reshape(ret.args[0], [int(x) for x in call.checked_type.shape])

        _broadcast_op = [
            relay.op.get("add"),
            relay.op.get("logical_and"),
            relay.op.get("logical_or"),
            relay.op.get("logical_xor"),
            relay.op.get("multiply"),
            relay.op.get("subtract"),
            relay.op.get("divide"),
            relay.op.get("mod"),
            relay.op.get("equal"),
            relay.op.get("not_equal"),
            relay.op.get("greater"),
            relay.op.get("greater_equal"),
            relay.op.get("less"),
            relay.op.get("less_equal"),
            relay.op.get("where"),
            relay.op.get("minimum"),
            relay.op.get("maximum"),
            relay.op.get("power"),
            relay.op.get("bitwise_and"),
            relay.op.get("bitwise_or"),
            relay.op.get("bitwise_xor"),
            relay.op.get("qnn.add"),
            relay.op.get("qnn.subtract"),
            relay.op.get("qnn.mul"),
        ]

        if ret.op in _broadcast_op:
            shape1 = ret.type_args[0].shape
            shape2 = ret.type_args[1].shape
            if shape1 != shape2:
                new_call = self._broadcast_inputs(ret)
                if new_call is not None:
                    return new_call

        if ret.op == relay.op.get("nn.upsampling"):
            attrs = ret.attrs
            layout = attrs.layout
            method = attrs.method
            align_corners = attrs.align_corners
            out_shape = call.checked_type.shape

            size = out_shape[layout.index("H") : layout.index("W") + 1]
            # convert "bilinear" and "bicubic" to "linear" and "cubic"
            method = method[2:] if method[:2] == "bi" else method
            coordinate_transformation_mode = "half_pixel"
            if align_corners:
                coordinate_transformation_mode = "align_corners"

            return relay.image.resize2d(
                ret.args[0], size, None, layout, method, coordinate_transformation_mode
            )

        if ret.op == relay.op.get("stack"):
            attrs = ret.attrs
            tensors = []

            # only one input convert to reshape.
            if len(ret.args[0]) == 1:
                return relay.reshape(ret.args[0][0], call.checked_type.shape)

            for i, t in enumerate(ret.args[0]):
                shape = call.args[0][i].checked_type.shape
                axis = len(shape) if attrs.axis == -1 else int(attrs.axis)
                new_shape = np.insert(np.array(shape), axis, 1)
                tensors.append(relay.reshape(t, new_shape))
            return relay.concatenate(tensors, axis)

        if ret.op == relay.op.get("reverse"):
            input_shape = [int(i) for i in call.args[0].checked_type.shape]
            if len(input_shape) > 3:
                return ret
            axis = ret.attrs.axis.value
            rev_axis = axis + len(input_shape) if axis < 0 else axis

            if len(input_shape) < 2:
                pre_reshape = relay.reshape(ret.args[0], [-1, 1])
                time_axis = 0
                batch = 1
                seq_len = input_shape[0]
                seq_lengths = np.array([seq_len] * batch, np.int32)
                reverse_sequence = relay.reverse_sequence(
                    pre_reshape,
                    relay.const(seq_lengths),
                    seq_axis=time_axis,
                    batch_axis=1 - time_axis,
                )
                return relay.reshape(reverse_sequence, input_shape)
            elif axis in (0, 1):
                time_axis = rev_axis
                batch = input_shape[1 - time_axis]
                seq_len = input_shape[time_axis]
                seq_lengths = np.array([seq_len] * batch, np.int32)
                return relay.reverse_sequence(
                    ret.args[0],
                    relay.const(seq_lengths),
                    seq_axis=time_axis,
                    batch_axis=1 - time_axis,
                )
            else:
                pre_perm = [rev_axis] + [idx for idx in range(len(input_shape)) if idx != rev_axis]
                pre_transpose = relay.transpose(ret.args[0], pre_perm)
                time_axis = 0
                batch = input_shape[0]
                seq_len = input_shape[rev_axis]
                seq_lengths = np.array([seq_len] * batch, np.int32)
                reverse_sequence = relay.reverse_sequence(
                    pre_transpose,
                    relay.const(seq_lengths),
                    seq_axis=time_axis,
                    batch_axis=1 - time_axis,
                )
                post_perm = [pre_perm.index(i) for i in range(len(pre_perm))]
                return relay.transpose(reverse_sequence, post_perm)

        if ret.op == relay.op.get("broadcast_to"):
            shape = ret.attrs.shape
            input_shape = call.args[0].checked_type.shape
            input_shape = [int(val) for val in input_shape]
            arg = ret.args[0]
            if len(shape) > len(input_shape):
                delta = len(shape) - len(input_shape)
                input_shape = [1] * delta + input_shape
                arg = relay.reshape(arg, input_shape)
            reps = [j if input_shape[i] == 1 else 1 for i, j in enumerate(shape)]
            if reps[0] == 1 and len(set(reps)) == 1:
                return ret.args[0]
            else:
                return relay.tile(arg, reps)

        if ret.op == relay.op.get("repeat"):
            attrs = ret.attrs
            axis = int(attrs.axis)
            repeats = int(attrs.repeats)
            input_shape = call.args[0].checked_type.shape
            dshape = list(input_shape)
            dim = len(dshape)
            if axis < 0:
                axis = axis + dim
            reps = [1] * dim
            reps[axis] = repeats
            tile = relay.tile(ret.args[0], reps)
            new_shape = dshape[:axis] + [repeats] + dshape[axis:]
            tile_reshape = relay.reshape(tile, new_shape)

            trans_idx = list(range(dim + 1))
            trans_idx[axis] = axis + 1
            trans_idx[axis + 1] = axis
            trans = relay.transpose(tile_reshape, trans_idx)

            dshape[axis] *= repeats
            reshape_back = relay.reshape(trans, dshape)
            return reshape_back

        if ret.op == relay.op.get("layout_transform"):
            support_convert = (
                ("NCHW32c", "NHWC"),
                ("NCHW16c", "NHWC"),
                ("NHWC", "NCHW32c"),
                ("NHWC", "NCHW16c"),
            )
            src_layout = ret.attrs.src_layout
            dst_layout = ret.attrs.dst_layout

            if (src_layout, dst_layout) in support_convert:
                if src_layout == "NHWC":
                    end_c = int(dst_layout[4:6])
                    in_shape = call.args[0].checked_type.shape
                    new_shape = in_shape[:-1] + [in_shape[-1] // end_c, end_c]
                    reshape = relay.reshape(ret.args[0], new_shape)
                    return relay.transpose(reshape, [0, 3, 1, 2, 4])
                else:
                    in_shape = call.args[0].checked_type.shape
                    new_shape = [in_shape[0], in_shape[2], in_shape[3], in_shape[1] * in_shape[4]]
                    transpose = relay.transpose(ret.args[0], [0, 2, 3, 1, 4])
                    return relay.reshape(transpose, new_shape)
            else:
                WARN(
                    f"Compass do not support layout transform from "
                    f"{src_layout} to {dst_layout}. "
                    f"It may be running on cpu."
                )

        if ret.op == relay.op.get("nn.adaptive_avg_pool1d"):

            in_shape = call.args[0].checked_type.shape
            out_shape = call.checked_type.shape

            strides = in_shape[1] // out_shape[1]
            kernel_size = in_shape[1] - (out_shape[1] - 1) * strides

            inp = relay.reshape(ret.args[0], [in_shape[0], 1, in_shape[1], in_shape[2]])
            outp = relay.nn.avg_pool2d(
                inp, (1, kernel_size), (1, strides), layout="NHWC", out_layout="NHWC"
            )

            return relay.reshape(outp, out_shape)

        if ret.op == relay.op.get("nn.adaptive_avg_pool2d"):
            attrs = ret.attrs
            input_shape = call.args[0].checked_type.shape
            assert attrs.layout == "NHWC" or attrs.layout == "NCHW"
            if attrs.layout == "NHWC":
                input_size = np.array([input_shape[1], input_shape[2]])
            else:
                input_size = np.array([input_shape[2], input_shape[3]])
            output_size = np.array(attrs.output_size)

            strides = input_size // output_size
            kernel_size = input_size - (output_size - 1) * strides

            return relay.nn.avg_pool2d(
                ret.args[0],
                tuple(kernel_size),
                tuple(strides),
                layout=attrs.layout,
                out_layout=attrs.out_layout,
            )

        _dense_matmul_op = [
            relay.op.get("nn.matmul"),
            relay.op.get("nn.dense"),
        ]

        if ret.op in _dense_matmul_op:
            if ret.op == relay.op.get("nn.dense"):
                if isinstance(ret.args[1], relay.Constant):
                    return ret
                transpose_a, transpose_b = False, True
            else:
                transpose_a, transpose_b = ret.attrs.transpose_a, ret.attrs.transpose_b
            inp1_shape = call.args[0].checked_type.shape
            inp2_shape = call.args[1].checked_type.shape
            inp1 = ret.args[0]
            inp2 = ret.args[1]
            dims = len(inp1_shape)
            attrs = ret.attrs

            inp2 = relay.reshape(inp2, [1] + [int(x) for x in inp2_shape])
            # like (32, 100) @ (100, 128), just convert to (1, 32, 100) @ (1, 100, 128)
            if dims == 2:
                inp1 = relay.reshape(inp1, [1] + [int(x) for x in inp1_shape])
                out = relay.nn.batch_matmul(inp1, inp2, attrs.out_dtype, transpose_a, transpose_b)
                out_shape = _infer_shape(out)
                return relay.reshape(out, [int(x) for x in out_shape[1:]])
            # like (3, 32, 100) @ (100, 128)
            elif dims == 3:
                return relay.nn.batch_matmul(inp1, inp2, attrs.out_dtype, transpose_a, transpose_b)
            # like (3, 4, 5, 32, 100) @ (100, 128)
            # first: (3 * 4 * 5, 32, 100) @ (1, 100, 128)
            # second: reshape to (3, 4, 5, 32, 128)
            else:
                batch = 1
                for i in range(dims - 2):
                    batch *= inp1_shape[i]

                inp1 = relay.reshape(inp1, [batch] + list(inp1_shape[-2:]))
                out = relay.nn.batch_matmul(inp1, inp2, attrs.out_dtype, transpose_a, transpose_b)

                dimn = inp1_shape[-2]
                unit = inp2_shape[-1]
                if transpose_a:
                    dimn = inp1_shape[-1]
                if transpose_b:
                    unit = inp2_shape[-2]
                return relay.reshape(out, list(inp1_shape[:-2]) + [dimn, unit])

        if ret.op == relay.op.get("sort"):
            attrs = ret.attrs
            axis = attrs.axis
            k = 0  # return all elements if k < 1.
            is_ascend = attrs.is_ascend
            topk = relay.topk(ret.args[0], k, axis, "both", is_ascend)

            # return topk value
            return topk[0]

        if ret.op == relay.op.get("nn.batch_flatten"):
            inp_shape = call.args[0].checked_type.shape
            return relay.reshape(call.args[0], [inp_shape[0], -1])

        if ret.op == relay.op.get("full"):
            if len(ret.attrs.shape) == 1:
                return ret

            # aipu tile support input >2dims only
            inp = relay.reshape(ret.args[0], [1] * len(ret.attrs.shape))
            return relay.tile(inp, ret.attrs.shape)

        if ret.op == relay.op.get("slice_like"):
            if self._is_dynamic(call.args[0]) or self._is_dynamic(call.args[1]):
                return ret

            src_shape = call.args[0].checked_type.shape
            target_shape = call.args[1].checked_type.shape
            begin_idx = [0] * len(src_shape)
            strides = [1] * len(src_shape)
            end_idx = [int(i) for i in src_shape]

            if ret.attrs.axes is None:
                for i, data in enumerate(src_shape):
                    if i < len(target_shape):
                        if target_shape[i] > data:
                            return ret
                        else:
                            end_idx[i] = target_shape[i]
            else:
                for axis in ret.attrs.axes:
                    axis = int(axis)
                    axis = axis if axis >= 0 else len(src_shape) + axis
                    if target_shape[axis] > src_shape[axis]:
                        return ret
                    else:
                        end_idx[axis] = target_shape[axis]
            return relay.strided_slice(ret.args[0], begin_idx, end_idx, strides)

        if ret.op == relay.op.get("take"):
            axis = ret.attrs.axis
            # The flattened input array is used when none axis.
            if axis is None:
                reshape = relay.op.reshape(ret.args[0], [-1])
                return relay.op.take(reshape, ret.args[1], axis=0)

        if ret.op == relay.op.get("nn.avg_pool2d"):
            attrs = ret.attrs
            padding = list(attrs.padding)
            has_negative_pad = any(p < 0 for p in padding)
            kernels = attrs.pool_size
            has_exceed_pad = any(p >= kernels[0] for p in padding[::2]) or any(
                p >= kernels[1] for p in padding[1::2]
            )
            if not has_negative_pad and not has_exceed_pad:
                return ret

            # extract negative pad
            inputs = ret.args[0]
            if has_negative_pad:
                tmp_padding = [pad if pad < 0 else 0 for pad in padding]
                pad_width = (
                    (0, 0),
                    (tmp_padding[0], tmp_padding[2]),
                    (tmp_padding[1], tmp_padding[3]),
                    (0, 0),
                )
                pad_node = relay.nn.pad(inputs, pad_width)
                padding = [pad if pad >= 0 else 0 for pad in padding]
                inputs = pad_node

            # extract exceed pad
            if has_exceed_pad:
                pad_width = ((0, 0), (padding[0], padding[2]), (padding[1], padding[3]), (0, 0))
                pad_node = relay.nn.pad(inputs, pad_width)
                padding = [0, 0, 0, 0]

            return relay.nn.avg_pool2d(
                pad_node,
                attrs.pool_size,
                attrs.strides,
                attrs.dilation,
                padding,
                attrs.layout,
                attrs.out_layout,
                attrs.ceil_mode,
                attrs.count_include_pad,
            )

        if ret.op == relay.op.get("nn.global_avg_pool2d"):
            input_shape = call.args[0].checked_type.shape
            attrs = ret.attrs
            input_size = np.array([input_shape[1], input_shape[2]])
            kernel_size = np.prod(input_size)
            # Currently do not support big kernel on 16bit.
            # So change to two avg pool.
            if kernel_size > 257:

                def _crack(k):
                    k = int(k)
                    for i in range(int(np.sqrt(k)), 0, -1):
                        if k % i != 0:
                            continue
                        return (k // i, i)

                crack_h = _crack(input_size[0])
                crack_w = _crack(input_size[1])
                first_kernel = (crack_h[0], crack_w[1])
                second_kernel = (crack_h[1], crack_w[0])
                if np.prod(first_kernel) > 256 or np.prod(second_kernel) > 256:
                    return ret
                first_pool = relay.nn.avg_pool2d(
                    ret.args[0], first_kernel, first_kernel, layout=attrs.layout
                )
                return relay.nn.avg_pool2d(
                    first_pool, second_kernel, second_kernel, layout=attrs.layout
                )

        if ret.op == relay.op.get("nn.dilate"):
            attrs = ret.attrs
            # [batch, spatial_shape, remaining_shape]
            # only support strides: [1, x, x, 1...] to convert b2s now
            strides = [int(s) for s in attrs.strides]
            stride_prod = np.prod(np.array(strides))
            if len(strides) <= 2 or stride_prod != strides[1] * strides[2]:
                return ret

            dilate_value = float(attrs.dilation_value)
            inp = ret.args[0]
            inp_shape = [int(dim) for dim in call.args[0].checked_type.shape]
            const_shape = [int(stride_prod) - 1] + inp_shape[1:]
            const = np.zeros(const_shape, dtype="float32") + dilate_value
            dilate_value_const = relay.const(const, dtype="float32")
            concat = relay.concatenate([inp, dilate_value_const], axis=0)
            block_shape = [strides[1], strides[2]]
            crops = [[0, strides[1] - 1], [0, strides[2] - 1]]
            out = relay.nn.batch_to_space_nd(concat, block_shape, crops)

            return out

        return ret


@relay.transform.function_pass(opt_level=0)
class ConvertAIPUOps:
    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        return AIPUOpsConvertor().visit(func)
