# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=invalid-name
"""Lower each virtual instruction to the composite of multiple real instructions."""
from tvm import ir, tir, DataType
from ... import script as S
from .utils import is_all_true_pred


class _Mutator(tir.StmtExprMutator):
    def _mutate_vmul(self, call):
        r, x, y, mask = call.args[1:]
        x_vdtype = DataType(x.dtype)

        if x_vdtype.is_float or (x_vdtype == DataType(y.dtype) and is_all_true_pred(mask)):
            # For float, there are equal-width multiply instructions.
            # For integer, Compass OpenCL C compiler support equal-width multiply only when the
            # multiply expression is represented through operator "*", i.e., mask must be all true.
            return call

        # There's any single real instruction for it, implement it through multiple instructions.
        out_sign = "s" if DataType(call.dtype).is_int else "u"
        if x_vdtype.bits == 16:
            # For i16x16/u16x16, the middle result type is i32x8/u32x8, so here
            # need cast the parameter R accordingly.
            lo_r, hi_r = S.vxtl(r), S.vxth(r)
        else:
            # For i32x8/u32x8, even through the middle result type looks like i32x8/u32x8, actually
            # they are i64x4/u64x4, so here need cast the parameter R accordingly too.
            lo_r, hi_r = S.vzip(r, r, "low"), S.vzip(r, r, "high")

        lo = S.vmull(x, y, mask=mask, out_sign=out_sign, r=lo_r)
        hi = S.vmulh(x, y, mask=mask, out_sign=out_sign, r=hi_r)

        if x_vdtype.bits == 16:
            # For i16x16/u16x16, the middle result type is i32x8/u32x8, so here
            # need reinterpret it to i16x16/u16x16.
            lo = S.reinterpret(lo, x_vdtype)
            hi = S.reinterpret(hi, x_vdtype)

        return S.vconcat(lo, hi, "even")

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
