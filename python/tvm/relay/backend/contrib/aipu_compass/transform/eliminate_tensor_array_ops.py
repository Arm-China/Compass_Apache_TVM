# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
# pylint: disable=bad-super-call
""" Rewrite module by pattern """
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    is_constant,
    wildcard,
)
from .hint_pattern_rewrite import RewriteDecorator


@RewriteDecorator
class EliminateTensorArrayRewriter(DFPatternCallback):
    """
    tensor + tensor_array_write + tensor_array_read +
        tensor_get_data => tensor
    tensor + tensor_array_unstack + tensor_array_scatter +
        tensor_array_read + tensor_get_data => tensor
    """

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.construct_op = wildcard()
        self.scatter_op = wildcard()
        self.array_read_op = wildcard()
        self.get_data_op = wildcard()

        self.inp = wildcard()
        self.constructed = self.construct_op(self.inp)
        self.scatter = self.scatter_op(wildcard(), is_constant(), self.constructed)
        self.array_read = self.array_read_op(self.scatter, is_constant())
        self.pattern = self.get_data_op(self.array_read)

    def callback(self, pre, post, node_map):
        scatter_op = node_map[self.scatter_op][0]
        array_read_op = node_map[self.array_read_op][0]
        get_data_op = node_map[self.get_data_op][0]

        if not isinstance(scatter_op, relay.GlobalVar):
            return post
        scatter_or_write = "scatter"

        if str(scatter_op.name_hint).startswith("tensor_array_scatter"):
            scatter_or_write = "scatter"
        elif str(scatter_op.name_hint).startswith("tensor_array_write"):
            scatter_or_write = "write"
        else:
            return post
        if not isinstance(array_read_op, relay.GlobalVar):
            return post
        if not str(array_read_op.name_hint).startswith("tensor_array_read"):
            return post
        if not isinstance(get_data_op, relay.GlobalVar):
            return post
        if not str(get_data_op.name_hint).startswith("tensor_get_data"):
            return post

        scatter = node_map[self.scatter][0]
        scatter_idx = scatter.args[1].data.numpy()
        array_read = node_map[self.array_read][0]
        read_idx = array_read.args[1].data.numpy()

        if int(scatter_idx) != int(read_idx):
            return post

        origin = node_map[self.inp][0]
        if scatter_or_write == "scatter":
            return relay.squeeze(origin, axis=[0])
        return origin


@tvm.ir.transform.module_pass(opt_level=0)
class EliminateTensorArrayOp:
    """
    Function to rewrite module by pattern
    """

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """try transform module"""

        fold_constant = relay.transform.FoldConstant()
        remove_unused = relay.transform.RemoveUnusedFunctions()

        pattern = EliminateTensorArrayRewriter(after_passes=[remove_unused, fold_constant])
        # pylint: disable=not-callable
        update_mod = pattern(mod)
        return update_mod
