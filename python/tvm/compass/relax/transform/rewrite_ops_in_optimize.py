# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Rewrite ops in optimize by mutator."""
from tvm import relax
from tvm.relax.expr_functor import PyExprMutator, mutator
from .utils import is_call
from ..utils import unpack_commutative_args


@mutator
class MergeAdjacentAdd(PyExprMutator):
    """Add(const) + Add(const) --> Add(const)."""

    def __init__(self, udchain, mod=None):
        super().__init__(mod)
        self.udchain = udchain

    def _check(self, call):
        if not is_call(call, "add"):
            return False
        if not any(isinstance(x, relax.Constant) for x in call.args):
            return False
        return True

    def visit_call_(self, call):
        post = super().visit_call_(call)
        if not self._check(post):
            return post
        inp, const0 = unpack_commutative_args(post)
        if inp not in self.udchain or len(self.udchain[inp]) > 1:
            return post
        inp_add = self.builder_.lookup_binding(inp)
        if not inp_add or not self._check(inp_add):
            return post
        inp0, const1 = unpack_commutative_args(inp_add)

        const0 = const0.data.numpy()
        const1 = const1.data.numpy()
        new_const = relax.const(const0 + const1, const0.dtype)
        return relax.op.add(inp0, new_const)


@relax.transform.function_pass(opt_level=0)
class RewriteOpsInOptimize:
    """Rewrite ops in optimize by mutator."""

    def transform_function(self, func, mod, pass_ctx):  # pylint: disable=unused-argument
        """Transform the given function and return the result."""
        if len(func.body.blocks) != 1:
            return func
        udchain = relax.analysis.udchain(func.body.blocks[0])
        updated_func = MergeAdjacentAdd(udchain).visit_expr(func)
        updated_func = relax.analysis.remove_all_unused(updated_func)

        return updated_func
