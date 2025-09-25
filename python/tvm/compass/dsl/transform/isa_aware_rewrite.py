# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Transform the TIR according to characteristics of each instruction."""
from tvm import ir, tir
from .. import script as S


class _Mutator(tir.StmtExprMutator):
    def _mutate_vabs(self, call):
        with_saturate = bool(call.args[4])
        ret_vdtype = call.dtype
        if with_saturate is True or not ret_vdtype.is_int:
            return call

        new_ret_vdtype = ret_vdtype.with_uint()
        new_args = list(call.args)
        new_args[1] = S.cast(new_args[1], new_ret_vdtype)
        return S.cast(tir.Call(new_ret_vdtype, call.op, new_args, call.span), ret_vdtype)

    def _mutate_vadd_vsub(self, call):
        if bool(call.args[5]) is True:  # With saturation.
            return call

        ret_vdtype = call.dtype
        new_args = list(call.args)
        for i in range(2, 4):
            if new_args[i].dtype != ret_vdtype:
                new_args[i] = S.cast(new_args[i], ret_vdtype)

        return tir.Call(call.dtype, call.op, new_args, call.span)

    def _mutate_vtbl(self, call):
        param_i = call.args[-1]
        dtype = param_i.dtype
        if dtype.is_uint:
            return call

        new_args = list(call.args)
        new_args[-1] = S.cast(param_i, dtype.with_uint())
        return tir.Call(call.dtype, call.op, new_args, call.span)

    def _mutate_vmull_vmulh(self, call):
        x, y = call.args[2], call.args[3]
        x_vdtype, y_vdtype = x.dtype, y.dtype
        ret_vdtype = call.dtype

        if ret_vdtype.is_uint and (x_vdtype.is_int or y_vdtype.is_int):
            new_ret_vdtype = ret_vdtype.with_int()
            new_args = list(call.args)
            new_args[1] = S.cast(new_args[1], new_ret_vdtype)
            return S.cast(tir.Call(new_ret_vdtype, call.op, new_args, call.span), ret_vdtype)
        if ret_vdtype.is_int and (x_vdtype.is_uint and y_vdtype.is_uint):
            new_ret_vdtype = ret_vdtype.with_uint()
            new_args = list(call.args)
            new_args[1] = S.cast(new_args[1], new_ret_vdtype)
            return S.cast(tir.Call(new_ret_vdtype, call.op, new_args, call.span), ret_vdtype)
        return call

    def _mutate_vsl(self, call):
        shift = call.args[3]
        shift_vdtype = shift.dtype

        if bool(call.args[5]) is True:  # With saturation.
            if shift_vdtype.is_int:
                return call

            new_args = list(call.args)
            new_args[3] = S.cast(shift, shift_vdtype.with_int())
            return tir.Call(call.dtype, call.op, new_args, call.span)

        x_vdtype = call.args[2].dtype
        if x_vdtype.type_code != shift_vdtype.type_code:
            new_args = list(call.args)
            new_shift_vdtype = shift_vdtype.with_type_code(x_vdtype.type_code)
            new_args[3] = S.cast(shift, new_shift_vdtype)
            return tir.Call(call.dtype, call.op, new_args, call.span)

        return call

    def _mutate_vnsr(self, call):
        shift = call.args[2]
        shift_vdtype = shift.dtype

        if shift_vdtype.is_int:
            new_args = list(call.args)
            new_args[2] = S.cast(shift, shift_vdtype.with_uint())
            return self.visit(tir.Call(call.dtype, call.op, new_args, call.span))

        with_saturate = bool(call.args[4])
        x_vdtype = call.args[1].dtype
        ret_vdtype = call.dtype
        if not with_saturate and x_vdtype.type_code != ret_vdtype.type_code:
            new_ret_vdtype = ret_vdtype.with_type_code(x_vdtype.type_code)
            new_call = tir.Call(new_ret_vdtype, call.op, call.args, call.span)
            return self.visit(S.cast(new_call, ret_vdtype))

        return call

    def _mutate_vload(self, call):
        ptr = call.args[1]

        if ptr.args[0].dtype != call.dtype:
            new_args = list(call.args)
            new_args[1] = tir.pointer(call.dtype, ptr.args[1], ptr, 0)
            return tir.Call(call.dtype, call.op, new_args, call.span)

        return call

    def _mutate_vstore(self, call):
        value = call.args[1]
        ptr = call.args[2]

        if ptr.args[0].dtype != value.dtype:
            new_args = list(call.args)
            new_args[2] = tir.pointer(value.dtype, ptr.args[1], ptr, 0)
            return tir.Call(call.dtype, call.op, new_args, call.span)

        return call

    def _mutate_vload_gather(self, call):
        ptr = call.args[1]

        if ptr.args[0].dtype != call.dtype:
            new_args = list(call.args)
            new_args[1] = tir.pointer(call.dtype, ptr.args[1], ptr, 0)
            return self.visit(tir.Call(call.dtype, call.op, new_args, call.span))

        offsets = call.args[2:-1]
        if any(not x.dtype.is_uint16 for x in offsets):
            new_args = list(call.args)
            for i, x in enumerate(offsets):
                new_args[2 + i] = S.cast(x, "uint16")
            return self.visit(tir.Call(call.dtype, call.op, new_args, call.span))

        return call

    def _mutate_vstore_scatter(self, call):
        value = call.args[1]
        ptr = call.args[2]

        if ptr.args[0].dtype != value.dtype:
            new_args = list(call.args)
            new_args[2] = tir.pointer(value.dtype, ptr.args[1], ptr, 0)
            return self.visit(tir.Call(call.dtype, call.op, new_args, call.span))

        offsets = call.args[3:-1]
        if any(not x.dtype.is_uint16 for x in offsets):
            new_args = list(call.args)
            for i, x in enumerate(offsets):
                new_args[3 + i] = S.cast(x, "uint16")
            return self.visit(tir.Call(call.dtype, call.op, new_args, call.span))

        return call

    def _mutate_vpcnt(self, call):
        x = call.args[2]
        x_vdtype = x.dtype
        if x_vdtype.is_uint:
            new_args = list(call.args)
            new_args[2] = S.cast(x, x_vdtype.with_int())
            return self.visit(tir.Call(call.dtype, call.op, new_args, call.span))

        ret_vdtype = call.dtype
        if ret_vdtype.is_int:
            new_ret_vdtype = ret_vdtype.with_uint()
            new_args = list(call.args)
            new_args[1] = S.cast(new_args[1], new_ret_vdtype)
            new_call = tir.Call(new_ret_vdtype, call.op, new_args, call.span)
            return self.visit(S.cast(new_call, ret_vdtype))

        return call

    def _mutate_vcls(self, call):
        ret_vdtype = call.dtype
        if ret_vdtype.is_int:
            new_ret_vdtype = ret_vdtype.with_uint()
            new_args = list(call.args)
            new_args[1] = S.cast(new_args[1], new_ret_vdtype)
            new_call = tir.Call(new_ret_vdtype, call.op, new_args, call.span)
            return self.visit(S.cast(new_call, ret_vdtype))

        return call

    def visit_call(self, call):
        ret = super().visit_call(call)

        if ret.op != ir.Op.get("tir.call_extern"):
            return ret

        func_name = ret.args[0].value
        if func_name == "vabs":
            return self._mutate_vabs(ret)
        elif func_name == "vadd":
            return self._mutate_vadd_vsub(ret)
        elif func_name == "vsub":
            return self._mutate_vadd_vsub(ret)
        elif func_name == "vtbl":
            return self._mutate_vtbl(ret)
        elif func_name in ("vmull", "vmulh"):
            return self._mutate_vmull_vmulh(ret)
        elif func_name == "vsl":
            return self._mutate_vsl(ret)
        elif func_name == "vnsr":
            return self._mutate_vnsr(ret)
        elif func_name == "vload":
            return self._mutate_vload(ret)
        elif func_name == "vstore":
            return self._mutate_vstore(ret)
        elif func_name == "vload_gather":
            return self._mutate_vload_gather(ret)
        elif func_name == "vstore_scatter":
            return self._mutate_vstore_scatter(ret)
        elif func_name == "__vpcnt":
            return self._mutate_vpcnt(ret)
        elif func_name == "__vcls":
            return self._mutate_vcls(ret)

        return ret


@tir.transform.prim_func_pass(opt_level=0)
class IsaAwareRewrite:
    """Transform the TIR according to characteristics of each instruction."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        return func.with_body(_Mutator().visit(func.body), span=func.span)
