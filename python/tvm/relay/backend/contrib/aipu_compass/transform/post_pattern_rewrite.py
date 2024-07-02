# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=bad-super-call, unsupported-binary-operation
""" Rewrite module by pattern """
import numpy as np
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    is_constant,
    is_op,
    wildcard,
)
from tvm.relay.op.contrib.aipu_compass.pattern_table import unpack_commutative_args
from .hint_pattern_rewrite import RewriteDecorator


@RewriteDecorator
class SimplifyAddZero(DFPatternCallback):
    """wildcard + add(zero constant) ===> wildcard"""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        add = is_op("add")(wildcard(), is_constant())
        self.pattern = add

    def callback(self, pre, post, node_map):
        add = node_map[self.pattern][0]
        if relay.ty.is_dynamic(add.checked_type):
            return post

        _, const = unpack_commutative_args(add)
        np_const = const.data.numpy()
        if np.allclose(np_const, 0):
            return add.args[0]

        return post


@tvm.ir.transform.module_pass(opt_level=0)
class PostPatternRewrite:
    """
    Function to rewrite module by pattern after pattern match
    """

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """try transform module"""

        update_mod = relay.transform.InferType()(mod)
        fold_constant = relay.transform.FoldConstant()

        patterns = [
            SimplifyAddZero(before_passes=fold_constant),
        ]
        for pattern in patterns:
            update_mod = pattern(update_mod)  # pylint: disable=not-callable

        return update_mod
