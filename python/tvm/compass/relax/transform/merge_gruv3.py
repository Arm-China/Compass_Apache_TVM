# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Merge gruv3 op."""
import numpy as np
from tvm import relax, ir
from tvm.relax.dpl import rewrite_call, wildcard, is_op, is_const, is_var, is_tuple_get_item, is_gv
from tvm.relax.dpl import is_tuple, is_if
from tvm.compass.relax import op as compass_op


class Merger:
    """Merge gruv3 op."""

    def __init__(self, mod):
        self.mod = mod
        self.inp = wildcard()
        reshape = is_op("relax.reshape")(self.inp, wildcard())
        reshape1 = is_op("relax.reshape")(reshape, wildcard())
        permute_dims = is_op("relax.permute_dims")(reshape1)
        scatter_nd = is_op("relax.scatter_nd")(is_const(), is_const(), permute_dims)
        self.init_state = is_const()
        self.candidate_weight = is_const()
        self.candidate_bias = is_const()
        self.gate_weight = is_const()
        self.gate_bias = is_const()
        self.loop_gvar = is_gv()
        call_loop = self.loop_gvar(
            is_const(),
            self.init_state,
            is_const(),
            self.candidate_bias,
            self.candidate_weight,
            self.gate_bias,
            self.gate_weight,
            scatter_nd,
        )
        tup_get_item = is_tuple_get_item(call_loop, 1)
        self.pattern = tup_get_item

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            loop_gv = matches[self.loop_gvar]
            if self.match_gruv3_loop(loop_gv, self.mod[loop_gv]):
                gru_inp = matches[self.inp]
                init_state = matches[self.init_state]
                candidate_w = matches[self.candidate_weight].data.numpy()
                candidate_b = matches[self.candidate_bias].data.numpy()
                gate_w = matches[self.gate_weight].data.numpy()
                gate_b = matches[self.gate_bias].data.numpy()
                weight = np.concatenate((gate_w, candidate_w), axis=1)
                bias = np.concatenate((gate_b, candidate_b), axis=0)
                weight = np.transpose(weight, (1, 0))
                weight = relax.const(weight, weight.dtype)
                bias = relax.const(bias, bias.dtype)
                return compass_op.gruv3(gru_inp, init_state, weight, bias, "Hn")
            return expr

        return self.pattern, rewriter

    def match_gruv3_loop(self, gvar, func):
        """Check if match gruv3 loop structure."""
        if len(func.body.blocks) != 2:
            return False
        if len(func.body.blocks[0].bindings) != 1 or len(func.body.blocks[1].bindings) != 1:
            return False

        time_loop_var = is_var()
        zeros_loop_var = is_var()
        strided_slice_loop_var = is_var()
        scatter_v3_loop_var = is_var()
        gate_kernel_loop_var = is_var()
        gate_bias_loop_var = is_var()
        candidate_kernel_loop_var = is_var()
        candidate_bias_loop_var = is_var()
        cond = is_op("relax.less")(time_loop_var, strided_slice_loop_var)

        gv4 = is_op("relax.take")(scatter_v3_loop_var, time_loop_var)
        gv5 = is_op("relax.concat")(is_tuple([gv4, zeros_loop_var]))
        gv6 = is_op("relax.matmul")(gv5, gate_kernel_loop_var)
        gv7 = is_op("relax.add")(gv6, gate_bias_loop_var)
        gv8 = is_op("relax.sigmoid")(gv7)
        gv9 = is_op("relax.split")(gv8)
        gv10 = is_tuple_get_item(gv9, 0)
        gv11 = is_tuple_get_item(gv9, 1)
        gv12 = is_op("relax.multiply")(gv10, zeros_loop_var)
        gv13 = is_op("relax.concat")(is_tuple([gv4, gv12]))
        gv14 = is_op("relax.matmul")(gv13, candidate_kernel_loop_var)
        gv15 = is_op("relax.add")(gv14, candidate_bias_loop_var)
        gv16 = is_op("relax.subtract")(is_const(), gv11)
        gv17 = is_op("relax.tanh")(gv15)
        gv18 = is_op("relax.multiply")(gv11, zeros_loop_var)
        gv19 = is_op("relax.multiply")(gv16, gv17)
        gv0 = is_op("relax.add")(time_loop_var, is_const())
        gv1 = is_op("relax.add")(gv18, gv19)
        true_pat = is_gv(gvar.name_hint)(
            gv0,
            gv1,
            strided_slice_loop_var,
            candidate_bias_loop_var,
            candidate_kernel_loop_var,
            gate_bias_loop_var,
            gate_kernel_loop_var,
            scatter_v3_loop_var,
        )

        false_pat = is_tuple(
            [
                time_loop_var,
                zeros_loop_var,
                strided_slice_loop_var,
                candidate_bias_loop_var,
                candidate_kernel_loop_var,
                gate_bias_loop_var,
                gate_kernel_loop_var,
                scatter_v3_loop_var,
            ]
        )

        var2val = relax.analysis.get_var2val(func)
        cond_expr = func.body.blocks[0].bindings[0].value
        func_if_expr = func.body.blocks[1].bindings[0].value
        true_expr = func_if_expr.true_branch.blocks[0].bindings[-1].value
        false_expr = func_if_expr.false_branch
        if_expr = relax.If(cond_expr, true_expr, false_expr)
        if_pat = is_if(cond, true_pat, false_pat)
        return if_pat.match(if_expr, var2val)


class TensorflowGruv3BothMerger:
    """Merge gruv3 op both merger."""

    def __init__(self, mod):
        self.mod = mod
        self.inp = wildcard()
        permute_dims = is_op("relax.permute_dims")(self.inp)
        scatter_nd = is_op("relax.scatter_nd")(is_const(), is_const(), permute_dims)
        self.init_state = is_const()
        self.candidate_weight = is_const()
        self.candidate_bias = is_const()
        self.gate_weight = is_const()
        self.gate_bias = is_const()
        self.loop_gvar = is_gv()
        call_loop = self.loop_gvar(
            is_const(),
            is_const(),
            is_const(),
            self.init_state,
            is_const(),
            is_const(),
            self.candidate_bias,
            self.candidate_weight,
            self.gate_bias,
            self.gate_weight,
            scatter_nd,
        )
        tup_get_item = is_tuple_get_item(call_loop, 2)
        take = is_op("relax.take")(tup_get_item, is_const())
        permute_dims1 = is_op("relax.permute_dims")(take)
        self.pattern = permute_dims1

    @property
    def pr(self):  # pylint: disable=invalid-name
        """Return pattern and rewriter."""

        def rewriter(expr, matches):
            loop_gv = matches[self.loop_gvar]
            if self.match_gruv3_loop(loop_gv, self.mod[loop_gv]):
                gru_inp = matches[self.inp]
                init_state = matches[self.init_state]
                candidate_w = matches[self.candidate_weight].data.numpy()
                candidate_b = matches[self.candidate_bias].data.numpy()
                gate_w = matches[self.gate_weight].data.numpy()
                gate_b = matches[self.gate_bias].data.numpy()
                weight = np.concatenate((gate_w, candidate_w), axis=1)
                bias = np.concatenate((gate_b, candidate_b), axis=0)
                weight = np.transpose(weight, (1, 0))
                weight = relax.const(weight, weight.dtype)
                bias = relax.const(bias, bias.dtype)
                return compass_op.gruv3(gru_inp, init_state, weight, bias, "H")
            return expr

        return self.pattern, rewriter

    def match_gruv3_loop(self, gvar, func):
        """Check if match gruv3 loop structure."""
        if len(func.body.blocks) != 2:
            return False
        if len(func.body.blocks[0].bindings) != 3 or len(func.body.blocks[1].bindings) != 1:
            return False

        while_loop_var_0 = is_var()
        time_loop_var = is_var()
        array_loop_var = is_var()
        zeros_loop_var = is_var()
        strided_slice_loop_var = is_var()
        minimum_loop_var = is_var()
        scatter_v3_loop_var = is_var()
        gate_kernel_loop_var = is_var()
        gate_bias_loop_var = is_var()
        candidate_kernel_loop_var = is_var()
        candidate_bias_loop_var = is_var()
        gv132 = is_op("relax.less")(while_loop_var_0, strided_slice_loop_var)
        gv133 = is_op("relax.less")(time_loop_var, minimum_loop_var)
        cond = is_op("relax.logical_and")(gv132, gv133)

        gv115 = is_op("relax.take")(scatter_v3_loop_var, time_loop_var)
        gv116 = is_op("relax.concat")(is_tuple([gv115, zeros_loop_var]))
        gv117 = is_op("relax.matmul")(gv116, gate_kernel_loop_var)
        gv118 = is_op("relax.add")(gv117, gate_bias_loop_var)
        gv119 = is_op("relax.sigmoid")(gv118)
        gv120 = is_op("relax.split")(gv119)
        gv121 = is_tuple_get_item(gv120, 0)
        gv122 = is_tuple_get_item(gv120, 1)
        gv123 = is_op("relax.multiply")(gv121, zeros_loop_var)
        gv124 = is_op("relax.concat")(is_tuple([gv115, gv123]))
        gv125 = is_op("relax.matmul")(gv124, candidate_kernel_loop_var)
        gv126 = is_op("relax.add")(gv125, candidate_bias_loop_var)
        gv127 = is_op("relax.subtract")(is_const(), gv122)
        gv128 = is_op("relax.tanh")(gv126)
        gv129 = is_op("relax.multiply")(gv122, zeros_loop_var)
        gv130 = is_op("relax.multiply")(gv127, gv128)
        gv134 = is_op("relax.reshape")(time_loop_var, wildcard())
        gv135 = is_op("relax.add")(gv129, gv130)
        gv1 = is_op("relax.add")(while_loop_var_0, is_const())
        gv2 = is_op("relax.add")(time_loop_var, is_const())
        gv3 = is_op("relax.scatter_nd")(array_loop_var, gv134, gv135)
        true_pat = is_gv(gvar.name_hint)(
            gv1,
            gv2,
            gv3,
            gv135,
            minimum_loop_var,
            strided_slice_loop_var,
            candidate_bias_loop_var,
            candidate_kernel_loop_var,
            gate_bias_loop_var,
            gate_kernel_loop_var,
            scatter_v3_loop_var,
        )

        false_pat = is_tuple(
            [
                while_loop_var_0,
                time_loop_var,
                array_loop_var,
                zeros_loop_var,
                minimum_loop_var,
                strided_slice_loop_var,
                candidate_bias_loop_var,
                candidate_kernel_loop_var,
                gate_bias_loop_var,
                gate_kernel_loop_var,
                scatter_v3_loop_var,
            ]
        )

        var2val = relax.analysis.get_var2val(func)
        cond_expr = func.body.blocks[0].bindings[2].value
        func_if_expr = func.body.blocks[1].bindings[0].value
        true_expr = func_if_expr.true_branch.blocks[0].bindings[-1].value
        false_expr = func_if_expr.false_branch
        if_expr = relax.If(cond_expr, true_expr, false_expr)
        if_pat = is_if(cond, true_pat, false_pat)
        return if_pat.match(if_expr, var2val)


@ir.transform.module_pass(opt_level=0)
class MergeGRUV3:
    """Merge gruv3 op."""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        opts = (TensorflowGruv3BothMerger(ir_mod), Merger(ir_mod))
        for opt in opts:
            for gvar, func in ir_mod.functions.items():
                ir_mod[gvar] = rewrite_call(*opt.pr, func)
        return relax.transform.RemoveUnusedOutputs()(ir_mod)
