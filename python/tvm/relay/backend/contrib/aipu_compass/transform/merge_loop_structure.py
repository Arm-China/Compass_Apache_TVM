# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Merge loop structure to global function calls"""
import numpy as np
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    FunctionPattern,
    is_constant,
    is_op,
    is_tuple_get_item,
    wildcard,
    is_var,
    is_tuple,
    is_if,
    is_let,
)
from .hint_pattern_rewrite import RewriteDecorator


@RewriteDecorator
class TensorflowGruv3Merger(DFPatternCallback):
    """Merge tensorflow gru v3 structure."""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)  # pylint: disable=bad-super-call
        self.while_loop_var_0 = is_var()
        self.time_loop_var = is_var()
        self.zeros_loop_var = is_var()
        self.strided_slice_1_loop_var = is_var()
        self.minimum_loop_var = is_var()
        self.scatter_v3_loop_var = is_var()
        self.gate_kernel_loop_var = is_var()
        self.gate_bias_loop_var = is_var()
        self.candidate_kernel_loop_var = is_var()
        self.candidate_bias_loop_var = is_var()
        self.while_loop = is_var()

        pattern0 = is_op("less")(self.while_loop_var_0, self.strided_slice_1_loop_var)
        pattern1 = is_op("less")(self.time_loop_var, self.minimum_loop_var)
        pattern = is_op("logical_and")(pattern0, pattern1)
        cond = is_op("min")(pattern)

        take = is_op("take")(self.time_loop_var, is_constant())
        read_array = wildcard()(self.scatter_v3_loop_var, take)
        get_data = wildcard()(read_array)
        concat_tuple = is_tuple((get_data, self.zeros_loop_var))
        concat = is_op("concatenate")(concat_tuple)
        gate_kernel = is_op("transpose")(self.gate_kernel_loop_var)
        gate = is_op("nn.dense")(concat, gate_kernel)
        gate = is_op("add")(gate, self.gate_bias_loop_var)
        gate = is_op("sigmoid")(gate)
        gate = is_op("split")(gate)

        zt_ = is_tuple_get_item(gate, 1)
        rt_ = is_tuple_get_item(gate, 0)
        rt_ = is_op("multiply")(rt_, self.zeros_loop_var)

        concat_1_tuple = is_tuple((get_data, rt_))
        concat_1 = is_op("concatenate")(concat_1_tuple)
        candidate_kernel = is_op("transpose")(self.candidate_kernel_loop_var)
        candicate = is_op("nn.dense")(concat_1, candidate_kernel)
        candicate = is_op("add")(candicate, self.candidate_bias_loop_var)

        sub = is_op("subtract")(is_constant(), zt_)
        gru_cell_tanh = is_op("tanh")(candicate)

        mul_1 = is_op("multiply")(zt_, self.zeros_loop_var)
        mul_2 = is_op("multiply")(sub, gru_cell_tanh)
        while_iter = is_op("add")(self.while_loop_var_0, is_constant())
        time_iter = is_op("add")(self.time_loop_var, is_constant())
        ht_ = is_op("add")(mul_1, mul_2)
        true_branch = self.while_loop(
            while_iter,
            time_iter,
            ht_,
            self.strided_slice_1_loop_var,
            self.minimum_loop_var,
            self.scatter_v3_loop_var,
            self.gate_kernel_loop_var,
            self.gate_bias_loop_var,
            self.candidate_kernel_loop_var,
            self.candidate_bias_loop_var,
        )
        false_branch = is_tuple(
            (
                self.while_loop_var_0,
                self.time_loop_var,
                self.zeros_loop_var,
                self.strided_slice_1_loop_var,
                self.minimum_loop_var,
                self.scatter_v3_loop_var,
                self.gate_kernel_loop_var,
                self.gate_bias_loop_var,
                self.candidate_kernel_loop_var,
                self.candidate_bias_loop_var,
            )
        )

        if_pattern = is_if(cond, true_branch, false_branch)
        let_func = FunctionPattern(
            [
                self.while_loop_var_0,
                self.time_loop_var,
                self.zeros_loop_var,
                self.strided_slice_1_loop_var,
                self.minimum_loop_var,
                self.scatter_v3_loop_var,
                self.gate_kernel_loop_var,
                self.gate_bias_loop_var,
                self.candidate_kernel_loop_var,
                self.candidate_bias_loop_var,
            ],
            if_pattern,
        )

        self.input = wildcard()
        transpose = is_op("transpose")(self.input)
        tensor_array_1 = wildcard()(is_constant())
        unstack = wildcard()(transpose)
        scatter = wildcard()(tensor_array_1, is_constant(), unstack)

        self.init_state = wildcard()
        self.candidate_weight = is_constant()
        self.candidate_bias = is_constant()
        self.gate_weight = is_constant()
        self.gate_bias = is_constant()
        end_call = self.while_loop(
            is_constant(),
            is_constant(),
            self.init_state,
            is_constant(),
            is_constant(),
            scatter,
            self.gate_weight,
            self.gate_bias,
            self.candidate_weight,
            self.candidate_bias,
        )

        self.pattern = is_let(self.while_loop, let_func, end_call)
        self.pattern = is_tuple_get_item(self.pattern, 2)
        self.pattern = is_op("reshape")(self.pattern)

    def callback(self, pre, post, node_map):
        gru_input = node_map[self.input][0]
        init_state = node_map[self.init_state][0]
        candidate_w = node_map[self.candidate_weight][0].data.numpy()
        candidate_b = node_map[self.candidate_bias][0].data.numpy()
        gate_w = node_map[self.gate_weight][0].data.numpy()
        gate_b = node_map[self.gate_bias][0].data.numpy()
        weight = np.concatenate((gate_w, candidate_w), axis=1)
        bias = np.concatenate((gate_b, candidate_b), axis=0)
        weight = np.transpose(weight, (1, 0))
        weight = relay.Constant(tvm.nd.array(weight))
        bias = relay.Constant(tvm.nd.array(bias))
        return relay.op.contrib.aipu_compass.gruv3(gru_input, init_state, weight, bias, "Hn")


@RewriteDecorator
class TensorflowGruv3BothMerger(DFPatternCallback):
    """Merge tensorflow gru v3 structure."""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)  # pylint: disable=bad-super-call
        self.while_loop_var_0 = is_var()
        self.time_loop_var = is_var()
        self.array_loop_var = is_var()
        self.zeros_loop_var = is_var()
        self.strided_slice_1_loop_var = is_var()
        self.minimum_loop_var = is_var()
        self.scatter_v3_loop_var = is_var()
        self.gate_kernel_loop_var = is_var()
        self.gate_bias_loop_var = is_var()
        self.candidate_kernel_loop_var = is_var()
        self.candidate_bias_loop_var = is_var()
        self.while_loop = is_var()

        pattern0 = is_op("less")(self.while_loop_var_0, self.strided_slice_1_loop_var)
        pattern1 = is_op("less")(self.time_loop_var, self.minimum_loop_var)
        pattern = is_op("logical_and")(pattern0, pattern1)
        cond = is_op("min")(pattern)
        take = is_op("take")(self.time_loop_var, is_constant())

        def cell_pattern(
            take_idx,
            zeros_loop_var,
            scatter_v3_loop_var,
            gate_kernel_loop_var,
            gate_bias_loop_var,
            candidate_kernel_loop_var,
            candidate_bias_loop_var,
        ):
            read_array = wildcard()(scatter_v3_loop_var, take_idx)
            get_data = wildcard()(read_array)
            concat_tuple = is_tuple((get_data, zeros_loop_var))
            concat = is_op("concatenate")(concat_tuple)
            gate_kernel = is_op("transpose")(gate_kernel_loop_var)
            gate = is_op("nn.dense")(concat, gate_kernel)
            gate = is_op("add")(gate, gate_bias_loop_var)
            gate = is_op("sigmoid")(gate)
            gate = is_op("split")(gate)

            zt_ = is_tuple_get_item(gate, 1)
            rt_ = is_tuple_get_item(gate, 0)
            rt_ = is_op("multiply")(rt_, zeros_loop_var)

            concat_1_tuple = is_tuple((get_data, rt_))
            concat_1 = is_op("concatenate")(concat_1_tuple)
            candidate_kernel = is_op("transpose")(candidate_kernel_loop_var)
            candicate = is_op("nn.dense")(concat_1, candidate_kernel)
            candicate = is_op("add")(candicate, candidate_bias_loop_var)

            sub = is_op("subtract")(is_constant(), zt_)
            gru_cell_tanh = is_op("tanh")(candicate)

            mul_1 = is_op("multiply")(zt_, zeros_loop_var)
            mul_2 = is_op("multiply")(sub, gru_cell_tanh)
            ht_ = is_op("add")(mul_1, mul_2)
            return ht_

        ht0 = cell_pattern(
            take,
            self.zeros_loop_var,
            self.scatter_v3_loop_var,
            self.gate_kernel_loop_var,
            self.gate_bias_loop_var,
            self.candidate_kernel_loop_var,
            self.candidate_bias_loop_var,
        )
        ht1 = cell_pattern(
            take,
            self.zeros_loop_var,
            self.scatter_v3_loop_var,
            self.gate_kernel_loop_var,
            self.gate_bias_loop_var,
            self.candidate_kernel_loop_var,
            self.candidate_bias_loop_var,
        )

        cur_array = wildcard()(ht0)
        while_iter = is_op("add")(self.while_loop_var_0, is_constant())
        time_iter = is_op("add")(self.time_loop_var, is_constant())
        array_write = wildcard()(self.array_loop_var, take, cur_array)

        true_branch = self.while_loop(
            while_iter,
            time_iter,
            array_write,
            ht1,
            self.strided_slice_1_loop_var,
            self.minimum_loop_var,
            self.scatter_v3_loop_var,
            self.gate_kernel_loop_var,
            self.gate_bias_loop_var,
            self.candidate_kernel_loop_var,
            self.candidate_bias_loop_var,
        )
        false_branch = is_tuple(
            (
                self.while_loop_var_0,
                self.time_loop_var,
                self.array_loop_var,
                self.zeros_loop_var,
                self.strided_slice_1_loop_var,
                self.minimum_loop_var,
                self.scatter_v3_loop_var,
                self.gate_kernel_loop_var,
                self.gate_bias_loop_var,
                self.candidate_kernel_loop_var,
                self.candidate_bias_loop_var,
            )
        )

        if_pattern = is_if(cond, true_branch, false_branch)
        func_pattern = FunctionPattern(
            [
                self.while_loop_var_0,
                self.time_loop_var,
                self.array_loop_var,
                self.zeros_loop_var,
                self.strided_slice_1_loop_var,
                self.minimum_loop_var,
                self.scatter_v3_loop_var,
                self.gate_kernel_loop_var,
                self.gate_bias_loop_var,
                self.candidate_kernel_loop_var,
                self.candidate_bias_loop_var,
            ],
            if_pattern,
        )

        self.input = wildcard()
        self.init_state = wildcard()
        self.gate_weight = is_constant()
        self.gate_bias = is_constant()
        self.candidate_weight = is_constant()
        self.candidate_bias = is_constant()

        transpose = is_op("transpose")(self.input)
        tensor_array_1 = wildcard()(is_constant())
        unstack = wildcard()(transpose)
        out_array = wildcard()(is_constant())
        scatter = wildcard()(tensor_array_1, is_constant(), unstack)

        end_call = self.while_loop(
            is_constant(),
            is_constant(),
            out_array,
            self.init_state,
            is_constant(),
            is_constant(),
            scatter,
            self.gate_weight,
            self.gate_bias,
            self.candidate_weight,
            self.candidate_bias,
        )
        self.pattern = is_let(self.while_loop, func_pattern, end_call)
        self.pattern = is_tuple_get_item(self.pattern, 2)

    def callback(self, pre, post, node_map):
        gru_input = node_map[self.input][0]
        init_state = node_map[self.init_state][0]
        candidate_w = node_map[self.candidate_weight][0].data.numpy()
        candidate_b = node_map[self.candidate_bias][0].data.numpy()
        gate_w = node_map[self.gate_weight][0].data.numpy()
        gate_b = node_map[self.gate_bias][0].data.numpy()
        weight = np.concatenate((gate_w, candidate_w), axis=1)
        bias = np.concatenate((gate_b, candidate_b), axis=0)
        weight = np.transpose(weight, (1, 0))
        weight = relay.Constant(tvm.nd.array(weight))
        bias = relay.Constant(tvm.nd.array(bias))
        return relay.op.contrib.aipu_compass.gruv3(gru_input, init_state, weight, bias, "H")


@RewriteDecorator
class TensorflowGrulMerger(DFPatternCallback):
    """Merge tensorflow gru_l structure."""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)  # pylint: disable=bad-super-call
        self.while_loop_var_0 = is_var()
        self.time_loop_var = is_var()
        self.zeros_loop_var = is_var()
        self.strided_slice_1_loop_var = is_var()
        self.scatter_v3_loop_var = is_var()
        self.gate_kernel_loop_var = is_var()
        self.gate_bias_loop_var = is_var()
        self.candidate_kernel_loop_var = is_var()
        self.candidate_bias_loop_var = is_var()
        self.while_loop = is_var()

        pattern = is_op("less")(self.time_loop_var, self.strided_slice_1_loop_var)
        cond = is_op("min")(pattern)

        take = is_op("take")(self.time_loop_var, is_constant())
        read_array = wildcard()(self.scatter_v3_loop_var, take)
        get_data = wildcard()(read_array)
        concat_tuple = is_tuple((get_data, self.zeros_loop_var))
        concat = is_op("concatenate")(concat_tuple)
        gate_kernel = is_op("transpose")(self.gate_kernel_loop_var)
        gate = is_op("nn.dense")(concat, gate_kernel)
        gate = is_op("add")(gate, self.gate_bias_loop_var)
        gate = is_op("sigmoid")(gate)
        gate = is_op("split")(gate)

        zt_ = is_tuple_get_item(gate, 1)
        rt_ = is_tuple_get_item(gate, 0)
        rt_ = is_op("multiply")(rt_, self.zeros_loop_var)

        concat_1_tuple = is_tuple((get_data, rt_))
        concat_1 = is_op("concatenate")(concat_1_tuple)
        candidate_kernel = is_op("transpose")(self.candidate_kernel_loop_var)
        candicate = is_op("nn.dense")(concat_1, candidate_kernel)
        candicate = is_op("add")(candicate, self.candidate_bias_loop_var)

        sub = is_op("subtract")(is_constant(), zt_)
        gru_cell_tanh = is_op("tanh")(candicate)

        mul_1 = is_op("multiply")(zt_, self.zeros_loop_var)
        mul_2 = is_op("multiply")(sub, gru_cell_tanh)
        time_iter = is_op("add")(self.time_loop_var, is_constant())
        ht_ = is_op("add")(mul_1, mul_2)
        true_branch = self.while_loop(
            time_iter,
            ht_,
            self.strided_slice_1_loop_var,
            self.scatter_v3_loop_var,
            self.gate_kernel_loop_var,
            self.gate_bias_loop_var,
            self.candidate_kernel_loop_var,
            self.candidate_bias_loop_var,
        )
        false_branch = is_tuple(
            (
                self.time_loop_var,
                self.zeros_loop_var,
                self.strided_slice_1_loop_var,
                self.scatter_v3_loop_var,
                self.gate_kernel_loop_var,
                self.gate_bias_loop_var,
                self.candidate_kernel_loop_var,
                self.candidate_bias_loop_var,
            )
        )

        if_pattern = is_if(cond, true_branch, false_branch)
        let_func = FunctionPattern(
            [
                self.time_loop_var,
                self.zeros_loop_var,
                self.strided_slice_1_loop_var,
                self.scatter_v3_loop_var,
                self.gate_kernel_loop_var,
                self.gate_bias_loop_var,
                self.candidate_kernel_loop_var,
                self.candidate_bias_loop_var,
            ],
            if_pattern,
        )

        self.input = is_var()
        reshape = is_op("reshape")(self.input)
        transpose = is_op("transpose")(reshape)
        tensor_array_1 = wildcard()(is_constant())
        unstack = wildcard()(transpose)
        scatter = wildcard()(tensor_array_1, is_constant(), unstack)

        self.init_state = is_constant()
        self.candidate_weight = is_constant()
        self.candidate_bias = is_constant()
        self.gate_weight = is_constant()
        self.gate_bias = is_constant()
        end_call = self.while_loop(
            is_constant(),
            self.init_state,
            is_constant(),
            scatter,
            self.gate_weight,
            self.gate_bias,
            self.candidate_weight,
            self.candidate_bias,
        )

        self.pattern = is_let(self.while_loop, let_func, end_call)
        self.pattern = is_tuple_get_item(self.pattern, 1)

    def callback(self, pre, post, node_map):
        gru_input = node_map[self.input][0]
        init_state = node_map[self.init_state][0]
        candidate_w = node_map[self.candidate_weight][0].data.numpy()
        candidate_b = node_map[self.candidate_bias][0].data.numpy()
        gate_w = node_map[self.gate_weight][0].data.numpy()
        gate_b = node_map[self.gate_bias][0].data.numpy()
        weight = np.concatenate((gate_w, candidate_w), axis=1)
        bias = np.concatenate((gate_b, candidate_b), axis=0)
        weight = np.transpose(weight, (1, 0))
        weight = relay.Constant(tvm.nd.array(weight))
        bias = relay.Constant(tvm.nd.array(bias))
        return relay.op.contrib.aipu_compass.gruv3(gru_input, init_state, weight, bias, "Hn")


@tvm.ir.transform.module_pass(opt_level=0)
class RemoveGRUv3ListReadAndGetPass:
    """After TensorflowGruv3BothMerger and if GRUv3 out_sequences is H,
    The pattern matches output is a List instead of tensors
    Usually a set of tensor_array_read/tensor_data_get is following,
    we need remove them.
    """

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """transform the module"""

        class RemoveGRUv3ListReadAndGet(relay.ExprMutator):
            """TensorflowGruv3BothMerger outputs is a list instead of tensors
            need remove the indefinite quantity tensor_array_read/tensor_get_data
            """

            def visit_call(self, call):
                post = super().visit_call(call)
                if post.op == relay.op.get("transpose"):
                    concat = post.args[0]
                    if isinstance(concat, relay.Call) and concat.op == relay.op.get("concatenate"):
                        tuple_len = len(concat.args[0].fields)
                        gruv3 = is_op("contrib.aipu_compass.gruv3")(
                            wildcard(), wildcard(), is_constant(), is_constant()
                        )
                        patterns = []
                        for _ in range(tuple_len):
                            read_array = wildcard()(gruv3, is_constant())
                            get_data = wildcard()(read_array)
                            expand = is_op("expand_dims")(get_data)
                            patterns.append(expand)
                        pattern = is_tuple(patterns)
                        if not pattern.match(post.args[0].args[0]):
                            return post
                        gruv3_expr = concat.args[0].fields[0].args[0].args[0].args[0]
                        return gruv3_expr
                return post

        mod["main"] = RemoveGRUv3ListReadAndGet().visit(mod["main"])
        return mod


@tvm.ir.transform.module_pass(opt_level=0)
class LoopStructureMerger:
    """merge loop structure to global function call"""

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """try transform module"""

        infer_type = relay.transform.InferType()
        patterns = [
            TensorflowGruv3BothMerger(after_passes=[RemoveGRUv3ListReadAndGetPass(), infer_type]),
            TensorflowGruv3Merger(after_passes=infer_type),
            TensorflowGrulMerger(after_passes=infer_type),
        ]

        for pattern in patterns:
            mod = pattern(mod)  # pylint: disable=not-callable
        return mod
