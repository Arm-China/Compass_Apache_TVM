# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""Sink operator "transpose" to make more optimization space."""
from tvm import ir, relay


class TransposeSinker(relay.ExprMutator):
    """Sinker of operator "transpose"."""

    def _convert_tgn_to_value(self, tgn):
        if isinstance(tgn, relay.TupleGetItem):
            return tgn.tuple_value
        return tgn

    def visit_call(self, call):
        post = super().visit_call(call)

        # Convert TupleGetItem to it's tuple value to get op.
        args = list(map(self._convert_tgn_to_value, post.args))

        if (
            post.op == relay.op.get("nn.relu")
            or post.op == relay.op.get("log")
            or post.op == relay.op.get("tanh")
        ) and (isinstance(args[0], relay.Call) and args[0].op == relay.op.get("transpose")):
            transpose = args[0]
            new_call = relay.Call(post.op, transpose.args, post.attrs, post.type_args, post.span)
            return relay.Call(
                transpose.op, [new_call], transpose.attrs, transpose.type_args, transpose.span
            )

        ops = [relay.op.get("add"), relay.op.get("multiply")]
        if post.op in ops:
            add = post
            # All operands are transpose.
            if (
                isinstance(args[0], relay.Call)
                and args[0].op == relay.op.get("transpose")
                and isinstance(args[1], relay.Call)
                and args[1].op == relay.op.get("transpose")
            ):
                transpose_a, transpose_b = add.args

                if not ir.structural_equal(transpose_a.attrs.axes, transpose_b.attrs.axes):
                    return add

                new_add = relay.Call(
                    add.op,
                    [transpose_a.args[0], transpose_b.args[0]],
                    add.attrs,
                    add.type_args,
                    add.span,
                )
                return relay.Call(
                    transpose_a.op,
                    [new_add],
                    transpose_a.attrs,
                    transpose_a.type_args,
                    transpose_a.span,
                )

            # One operand is transpose and the other is constant.
            if (
                isinstance(args[0], relay.Call)
                and args[0].op == relay.op.get("transpose")
                and isinstance(args[1], relay.Constant)
            ) or (
                isinstance(args[0], relay.Constant)
                and isinstance(args[1], relay.Call)
                and args[1].op == relay.op.get("transpose")
            ):
                if isinstance(args[0], relay.Constant):
                    constant = args[0]
                    constant_ndim = len(call.args[0].checked_type.shape)
                    transpose = args[1]
                else:
                    constant = args[1]
                    constant_ndim = len(call.args[1].checked_type.shape)
                    transpose = args[0]

                if constant_ndim != len(transpose.attrs.axes):
                    constant_val = constant.data.numpy()
                    broadcast_shape = [1] * (len(transpose.attrs.axes) - constant_ndim)
                    broadcast_shape = broadcast_shape + list(constant_val.shape)
                    constant = relay.reshape(constant, broadcast_shape)

                axes_dict = {axis: i for i, axis in enumerate(transpose.attrs.axes)}
                ordered_d = dict(sorted(axes_dict.items()))
                inverse_axes = list(ordered_d.values())
                new_constant = relay.transpose(constant, inverse_axes)
                new_add = relay.Call(
                    add.op,
                    [transpose.args[0], new_constant],
                    add.attrs,
                    add.type_args,
                    add.span,
                )
                return relay.Call(
                    transpose.op,
                    [new_add],
                    transpose.attrs,
                    transpose.type_args,
                    transpose.span,
                )

        if (
            post.op == relay.op.get("mean")
            and isinstance(args[0], relay.Call)
            and args[0].op == relay.op.get("transpose")
        ):
            mean = post
            transpose = mean.args[0]
            new_axes = []
            for x in mean.attrs.axis:
                new_axes.append(transpose.attrs.axes[x.value])

            new_attrs = {str(k): mean.attrs[k] for k in mean.attrs.keys()}
            new_attrs["axis"] = new_axes
            new_attrs = ir.make_node(str(mean.attrs).split("(")[0], **new_attrs)
            new_mean = relay.Call(mean.op, transpose.args, new_attrs, mean.type_args, mean.span)
            return relay.Call(
                transpose.op, [new_mean], transpose.attrs, transpose.type_args, transpose.span
            )

        if (
            post.op == relay.op.get("nn.lrn")
            and isinstance(args[0], relay.Call)
            and args[0].op == relay.op.get("transpose")
        ):
            lrn = post
            transpose = lrn.args[0]
            new_attrs = {str(k): lrn.attrs[k] for k in lrn.attrs.keys()}
            new_attrs["axis"] = transpose.attrs.axes[lrn.attrs.axis]
            new_attrs = ir.make_node(str(lrn.attrs).split("(")[0], **new_attrs)
            new_lrn = relay.Call(lrn.op, transpose.args, new_attrs, lrn.type_args, lrn.span)
            return relay.Call(
                transpose.op, [new_lrn], transpose.attrs, transpose.type_args, transpose.span
            )

        if post.op == relay.op.get("nn.prelu"):
            if (
                isinstance(args[0], relay.Call)
                and args[0].op == relay.op.get("transpose")
                and isinstance(args[1], relay.Constant)
            ):
                prelu = post
                transpose = post.args[0]
                alpha = post.args[1]

                new_axes = transpose.attrs.axes[prelu.attrs.axis]
                new_attrs = {"axis": new_axes}
                new_attrs = ir.make_node(str(prelu.attrs).split("(")[0], **new_attrs)
                new_prelu = relay.Call(
                    prelu.op, [transpose.args[0], alpha], new_attrs, prelu.type_args, prelu.span
                )
                return relay.Call(
                    transpose.op, [new_prelu], transpose.attrs, transpose.type_args, transpose.span
                )

        if post.op == relay.op.get("transpose"):
            if (
                isinstance(args[0], relay.Call)
                and args[0].op == relay.op.get("expand_dims")
                and isinstance(args[0].args[0], relay.Call)
                and args[0].args[0].op == relay.op.get("transpose")
                and not relay.ty.is_dynamic(call.checked_type)
            ):
                axis = args[0].attrs.axis
                axes = post.attrs.axes
                new_axes = []
                for i in axes:
                    if i < axis:
                        new_axes.append(i)
                    if i > axis:
                        new_axes.append(i - 1)
                transpose = relay.transpose(args[0].args[0], new_axes)
                return relay.reshape(transpose, call.checked_type.shape)
        return post


@relay.transform.function_pass(opt_level=0)
class SinkTranspose:
    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        return TransposeSinker().visit(func)
