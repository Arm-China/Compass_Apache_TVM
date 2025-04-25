# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=too-many-nested-blocks
"""Convert ops to equivalent but more efficient ops."""
import numpy as np
from tvm import relax, ir
from tvm.relax.expr_functor import PyExprMutator, mutator


@mutator
class Convertor(PyExprMutator):
    """Convert ops to equivalent but more efficient ops."""

    def _transpose2reshape(self, ret):
        if ret.op == ir.Op.get("relax.permute_dims"):
            out_shape = [int(x) for x in ret.struct_info.shape]
            for dim in out_shape:
                if dim != 1 and dim != np.prod(out_shape):
                    return ret
            return relax.op.reshape(ret.args[0], out_shape)
        return ret

    def visit_call_(self, call):
        ret = super().visit_call_(call)
        ret = self._transpose2reshape(ret)
        return ret


@ir.transform.module_pass(opt_level=0)
class ConvertOps:
    """Function to rewrite module by pattern"""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        ir_mod["main"] = Convertor().visit_expr(ir_mod["main"])
        return ir_mod
