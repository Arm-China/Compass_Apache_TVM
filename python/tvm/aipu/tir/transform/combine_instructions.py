# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Replaces a sequence of simple instructions with an equivalent optimized instruction."""
from tvm import ir, tir, DataType
from ... import script as S
from .utils import is_builtin, is_all_true_pred, is_const_pred, is_low_true_pred


def _try_expand_mask(mask, lanes, part):
    lanes_diff = lanes - DataType(mask.dtype).lanes

    if is_const_pred(mask):
        bool_arr = list(mask.args)
        extra_arr = [True] * lanes_diff
        new_bool_arr = (bool_arr + extra_arr) if part == "low" else (extra_arr + bool_arr)
        return tir.const_pred(new_bool_arr, mask.span)

    if is_low_true_pred(mask):
        new_n = mask.args[0] if part == "low" else (mask.args[0] + lanes_diff)
        return tir.low_true_pred(new_n, lanes, mask.span)

    return None


class _Mutator(tir.StmtExprMutator):
    def _try_pattern_vnsr(self, vcast):
        if vcast.args[0].value != "vcast":
            return None

        _, part, saturate, *vcast_inputs = vcast.args
        from_dtype, to_dtype = DataType(vcast_inputs[0].dtype), DataType(vcast.dtype)
        if not (from_dtype.is_integer and to_dtype.is_integer and from_dtype.bits > to_dtype.bits):
            return None

        saturate = bool(saturate)
        out_sign = "s" if to_dtype.is_int else "u"
        to_h = to_dtype.bits == 16
        vnsrs = []
        for vsr in vcast_inputs:
            if not (is_builtin(vsr, "vsr") and is_all_true_pred(vsr.args[4])):
                return None  # "vnsr" haven't parameter "r", so only can support all true mask.

            (x, shift), with_round = vsr.args[2:4], bool(vsr.args[5])
            vnsrs.append(S.vnsr(x, shift, None, saturate, out_sign, with_round, to_h))

        # If it can get here, it means that this pattern has been matched.
        input_cnt = len(vnsrs)

        # 1. Cast to narrower bits with merge, e.g., (i16x16, i16x16) -> i8x32,
        #    (i32x8, i32x8, i32x8) -> i8x32, (i32x8, i32x8) -> u16x16.
        if input_cnt > 1:
            assert part == "all", f'Unsupported cast that merge to "{to_dtype}" with part "{part}".'
            msg = f'Unsupported cast from {(from_dtype,) * input_cnt} to "{to_dtype}".'

            # 1.1 From 16-bit to 8-bit or 32-bit to 16-bit, e.g., (i16x16, i16x16) -> i8x32,
            #     (i32x8, i32x8) -> u16x16.
            if (from_dtype.bits, to_dtype.bits) in ((16, 8), (32, 16)):
                assert input_cnt == 2, msg
                return S.vconcat(vnsrs[0], vnsrs[1], "even")

            # 1.2 From 32-bit to 8-bit.
            assert (from_dtype.bits, to_dtype.bits) == (32, 8), msg
            low_16x16 = S.vconcat(vnsrs[0], vnsrs[1], "even")

            if input_cnt == 2:  # e.g., (i32x8, i32x8) -> i8x32.
                useless = low_16x16
                return S.vconcat(low_16x16, useless, "even")
            if input_cnt == 3:  # e.g., (i32x8, i32x8, i32x8) -> i8x32.
                useless = vnsrs[2]
                return S.vconcat(low_16x16, S.vconcat(vnsrs[2], useless, "even"), "even")

            assert input_cnt == 4, msg  # e.g., (i32x8, i32x8, i32x8, i32x8) -> i8x32.
            return S.vconcat(low_16x16, S.vconcat(vnsrs[2], vnsrs[3], "even"), "even")

        # 2. Cast to narrower bits without merge, e.g., i32 -> i8, i16 -> u8.
        msg = f'Unsupported cast from "{from_dtype}" to "{to_dtype}" with part "{part}".'
        assert part == "all", msg
        # 2.1 From 16-bit to 8-bit or 32-bit to 16-bit, e.g., i16x16 -> i8x32, i32x8 -> u16x16.
        if (from_dtype.bits, to_dtype.bits) in ((16, 8), (32, 16)):
            useless = vnsrs[0]
            return S.vconcat(vnsrs[0], useless, "even")

        # 2.2 From 32-bit to 8-bit, e.g., i32x8 -> i8x32, u32x8 -> i8x32.
        assert (from_dtype.bits, to_dtype.bits) == (32, 8), msg
        return S.vcompt(vnsrs[0], "8TFFF")

    def _try_pattern_vmull_vmulh(self, vmul):
        if vmul.args[0].value != "vmul":
            return None

        _, r, vcast_x, vcast_y, mask = vmul.args
        if not is_builtin(vcast_x, "vcast") or not is_builtin(vcast_y, "vcast"):
            return None

        if not (len(vcast_x.args) == len(vcast_y.args) == 4):
            return None  # Indicate it will cast to narrower bits with merge.

        _, part_x, saturate_x, x = vcast_x.args
        _, part_y, saturate_y, y = vcast_y.args
        vmul_dtype = DataType(vmul.dtype)
        x_dtype, y_dtype = DataType(x.dtype), DataType(y.dtype)

        if not (vmul_dtype.is_integer and x_dtype.is_integer and y_dtype.is_integer):
            return None

        # From here, both input and return data type must be integer.
        if saturate_x or saturate_y:
            return None

        if not (x_dtype.bits == y_dtype.bits and x_dtype.bits < vmul_dtype.bits):
            return None

        if not (part_x == part_y and part_x not in ("even", "odd")):
            return None

        # If it can get here, it means that this pattern has been matched.
        part = part_x
        if x_dtype.bits == 8 and vmul_dtype.bits == 32:
            part_for_16bit = "low" if part in ("ll", "lh") else "high"
            x = S.cast(x, x_dtype.with_bits(16).element_of, part=part_for_16bit)
            y = S.cast(y, y_dtype.with_bits(16).element_of, part=part_for_16bit)
            x_dtype, y_dtype = DataType(x.dtype), DataType(y.dtype)
            part = "low" if part in ("ll", "hl") else "high"

        vmull_or_vmulh = S.vmull if part == "low" else S.vmulh
        new_mask = _try_expand_mask(mask, x_dtype.lanes, part)
        out_sign = "s" if vmul_dtype.is_int else "u"

        if new_mask is None:
            # Indicate "mask" is a variable and can't be expanded during compilation time, and "r"
            # must be a variable too, it is guaranteed by parser.
            return S.vsel(vmull_or_vmulh(x, y, out_sign=out_sign), r, mask)

        # The "mask" is a "tir.const_pred" or "tir.low_true_pred" node, have been expanded
        # successfully, e.g. boolx8 -> boolx16.
        return vmull_or_vmulh(x, y, mask, out_sign=out_sign, r=r)

    def visit_call(self, call):
        ret = super().visit_call(call)

        if ret.op != ir.Op.get("tir.call_extern"):
            return ret

        combine_ret = self._try_pattern_vnsr(ret)
        if combine_ret is not None:
            return self.visit(combine_ret)

        combine_ret = self._try_pattern_vmull_vmulh(ret)
        if combine_ret is not None:
            return self.visit(combine_ret)

        return ret


@tir.transform.prim_func_pass(opt_level=0)
class CombineInstructions:
    """Replaces a sequence of simple instructions with an equivalent optimized instruction."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        return func.with_body(_Mutator().visit(func.body), span=func.span)
