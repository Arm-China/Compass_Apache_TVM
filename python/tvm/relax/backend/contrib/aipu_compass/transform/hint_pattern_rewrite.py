# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=too-many-nested-blocks
""" Rewrite module by pattern """
from tvm import relax, ir
from tvm.tir import IntImm
from tvm.relax.dpl import is_op, rewrite_call, wildcard, is_const


class MergeConsecutiveReshape:
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


class FuseConstToConvWeight:
    """
    src:        [(WX - S) / D] * M + A
    simplify:   (WM / D) * X + (A - (SM / D)
    new_weight: (WM / D), new_add: (A - (SM / D)
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
            sub_const = matches[self.sub].args[1].data.numpy().reshape(-1, 1, 1, 1)
            div_const = matches[self.div].args[1].data.numpy().reshape(-1, 1, 1, 1)
            mul_const = matches[self.mul].args[1].data.numpy().reshape(-1, 1, 1, 1)
            add = matches[self.add]
            add_const = matches[self.add].args[1].data.numpy()
            add_const_shape = add_const.shape
            add_const = add_const.reshape(-1, 1, 1, 1)

            if sub_const.shape != div_const.shape != mul_const.shape != add_const.shape:
                return expr
            # Weight layout: OIHW, ensure const_num == 'O'_num
            if conv.attrs.kernel_layout != "OIHW" or sub_const.shape[0] != weight.shape[0]:
                return expr

            new_weight_data = weight * mul_const / div_const
            new_add_const_data = add_const - sub_const * mul_const / div_const
            new_add_const_data = new_add_const_data.reshape(add_const_shape)
            new_weight = relax.const(new_weight_data, str(new_weight_data.dtype))
            new_add_const = relax.const(new_add_const_data, str(new_add_const_data.dtype))
            new_conv = relax.Call(conv.op, [conv.args[0], new_weight], conv.attrs)
            new_add = relax.Call(add.op, [new_conv, new_add_const], add.attrs)

            return new_add

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


@ir.transform.module_pass(opt_level=0)
class HintPatternRewrite:
    """Function to rewrite module by pattern"""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        opts = (
            FuseConstToConvWeight(),
            MergeConsecutiveReshape(),
            EliminateUselessPermuteDims(),
            MergeQuantCast(),
            MergePermMean(),
        )

        updated_func = ir_mod["main"]
        for opt in opts:
            updated_func = rewrite_call(*opt.pr, updated_func)
        ir_mod["main"] = updated_func

        return ir_mod
