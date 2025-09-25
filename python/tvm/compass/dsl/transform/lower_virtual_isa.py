# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=invalid-name
"""Lower each virtual instruction to the composite of multiple real instructions."""
from tvm import ir, tir
from .. import script as S
from .utils import is_all_true_pred


class _Mutator(tir.StmtExprMutator):
    def _mutate_vmul(self, call):
        r, x, y, mask = call.args[1:]
        x_vdtype = x.dtype

        if x_vdtype.is_floating or (x_vdtype == y.dtype and is_all_true_pred(mask)):
            # For float, there are equal-width multiply instructions.
            # For integer, Compass OpenCL C compiler support equal-width multiply only when the
            # multiply expression is represented through operator "*", i.e., mask must be all true.
            return call

        # There's any single real instruction for it, implement it through multiple instructions.
        out_sign = "s" if call.dtype.is_int else "u"
        new_vmul = S.vmul(x, y, out_sign=out_sign)
        return S.vsel(new_vmul, r, mask)

    def visit_call(self, call):
        ret = super().visit_call(call)

        if ret.op != ir.Op.get("tir.call_extern"):
            return ret

        func_name = ret.args[0].value
        if func_name == "vmul":
            return self._mutate_vmul(ret)

        return ret


@tir.transform.prim_func_pass(opt_level=0)
class LowerVirtualIsa:
    """Lower each virtual instruction to the composite of multiple real instructions."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        return func.with_body(_Mutator().visit(func.body), span=func.span)
