# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Align the width of all wider vector nodes with the hardware vector width by split."""
import math
import numpy as np
from tvm import tir, ir
from ..analysis import get_mask_associated_dtype
from ..utils import hw_native_vdtype, is_pointer, is_type_annotation
from .. import script as S
from .utils import is_builtin


_default_funcs = ("vclt", "vcgt", "vcle", "vcge", "vceq", "vcneq", "vxor")
_default_funcs += ("vadd", "vsub", "vmul", "__vdiv", "vexp", "vtanh", "vlog", "__vrint", "vabs")
_default_funcs += ("__vsel", "vpow", "__vfma", "vsin", "vcos", "vrsqrt", "vsqrt", "vfloor", "vceil")
_default_funcs += ("vsl", "vsr", "vror", "__vcls", "__vclz", "__vmod", "vand", "vor", "vinv")
_default_funcs += ("__vbrevs", "__vpcnt", "__vmax", "__vmin", "__vclass", "vnsr", "__vdpa")
_default_funcs += ("__vqdpa", "__vdot", "__vqdot", "verf", "vexp2", "vmul_hi")


def _move_pointer_in_scalar_unit(ptr, offset):
    lanes = ptr.args[0].dtype.lanes
    assert offset % lanes == 0
    new_args = list(ptr.args)
    new_args[3] += offset // lanes
    return tir.Call(ptr.dtype, ptr.op, new_args, ptr.span)


def _ceil_to_multiple_of_8(x):
    if x <= 8:
        return 8
    if x <= 16:
        return 16
    assert x <= 32
    return 32


def _vshfl(x, shift):
    return tir.call_extern(hw_native_vdtype(x.dtype), "__vshfl", x, shift)


def _vload(ptr, lanes):
    stride, mask = 1, tir.const_pred((True,) * lanes)
    return tir.call_extern(ptr.dtype.with_lanes(lanes), "vload", ptr, stride, mask)


def _vstore(value, ptr, stride=1, mask=None):
    mask = tir.const_pred((True,) * value.dtype.lanes) if mask is None else mask
    return tir.call_extern("void", "vstore", value, ptr, stride, mask)


def _split_half_bool(mask):
    dtype = mask.dtype.with_lanes(mask.dtype.lanes // 2)
    bool_l = tir.call_extern(dtype, "__vbxul", mask)
    bool_h = tir.call_extern(dtype, "__vbxuh", mask)
    return bool_l, bool_h


def _split_quarter_bool(mask):
    bool16_l, bool16_h = _split_half_bool(mask)
    bool8_ll, bool8_lh = _split_half_bool(bool16_l)
    bool8_hl, bool8_hh = _split_half_bool(bool16_h)
    return bool8_ll, bool8_lh, bool8_hl, bool8_hh


def _get_new_bool(masks, part_cnt):
    cnt = len(masks)
    if cnt > part_cnt:
        if cnt // part_cnt == 2:
            return tuple(S.vconcat(masks[i : i + 2]) for i in range(0, cnt, 2))
        if cnt // part_cnt == 4:
            return tuple(S.vconcat(masks[i : i + 4]) for i in range(0, cnt, 4))
    elif cnt < part_cnt:
        if part_cnt // cnt == 2:
            return tuple(x for mask in masks for x in _split_half_bool(mask))
        if part_cnt // cnt == 4:
            return tuple(x for mask in masks for x in _split_quarter_bool(mask))
    else:
        return masks

    raise RuntimeError(
        f"Unexpected split part number for len(mask) is {cnt} and part_cnt is {part_cnt}."
    )


def _get_part_cnt(args, part_cnt):
    result = []
    index_map = []
    for i, arg in enumerate(args):
        # Deal with pointer.
        if is_pointer(arg) or is_type_annotation(arg):
            continue
        if isinstance(arg, (tir.Call, tir.Var)):
            result.append(part_cnt)
            index_map.append(i)
        elif isinstance(arg, tuple):
            result.append(len(arg))
            index_map.append(i)

    result = result or [part_cnt]
    return result, index_map


class _Mutator(tir.StmtExprMutator):
    def __init__(self, cps_info, mask2associated_dtype, var_substitute_map):
        super().__init__()
        self._cps_info = cps_info
        self._mask2associated_dtype = mask2associated_dtype
        self._var_substitute_map = var_substitute_map
        self._var2buffer = {}

    def visit_var(self, var):
        return self._var_substitute_map.get(var, var)

    def _get_associated_dtype(self, call):
        dtype = call.dtype
        assert dtype.is_bool, "Only can get associated dtype through boolean data type."
        if call in self._mask2associated_dtype:
            return self._mask2associated_dtype[call]
        return dtype.with_bits(self._cps_info.vector_width // dtype.lanes)

    def _get_hw_lanes(self, dtype):
        assert not dtype.is_bool, "Can't get hardware lanes through boolean data type."
        return self._cps_info.vector_width // dtype.bits

    def _mutate_args(self, args, part_cnt=1):
        ret_args = list(self.visit_expr(x) for x in args)

        part_cnts, index_map = _get_part_cnt(ret_args, part_cnt)
        if len(set(part_cnts)) != 1:
            # Here use 0 because the mask is always the last parameter.
            part_cnt = part_cnts[0]
            mask_indices = [index_map[i] for i, value in enumerate(part_cnts) if value != part_cnt]
            for i in mask_indices:
                masks = (ret_args[i],) if not isinstance(ret_args[i], tuple) else ret_args[i]
                msg = "Only boolean data type variable can be update during mutate args."
                assert all(mask.dtype.is_bool for mask in masks), msg
                ret_args[i] = _get_new_bool(masks, part_cnt)
        else:
            part_cnt = part_cnts.pop()

        ret_args = tuple(x if isinstance(x, tuple) else (x,) * part_cnt for x in ret_args)
        return tuple(list(x) for x in zip(*ret_args))

    def _mutate_const_pred(self, call):
        dtype = call.dtype
        associated_dtype = self._get_associated_dtype(call)
        assert dtype.lanes <= associated_dtype.lanes, 'Error in "get_mask_associated_dtype".'
        hw_lanes = self._get_hw_lanes(associated_dtype)

        if associated_dtype.lanes <= hw_lanes:
            return super().visit_call(call)

        # Need split to multiple parts according to the hardware vector width used by the user.
        part_cnt = math.ceil(associated_dtype.lanes / hw_lanes)
        bool_arr = list(call.args) + [False] * (associated_dtype.lanes - dtype.lanes)

        ret = []
        for i in range(part_cnt):
            cur_lanes = min(associated_dtype.lanes - i * hw_lanes, hw_lanes)
            new_args = bool_arr[i * hw_lanes : i * hw_lanes + cur_lanes]
            ret.append(tir.Call(dtype.with_lanes(cur_lanes), call.op, new_args, call.span))

        return tuple(ret)

    def _mutate_low_true_pred(self, call):
        dtype = call.dtype
        hw_lanes = self._get_hw_lanes(self._get_associated_dtype(call))

        if dtype.lanes <= hw_lanes:
            return super().visit_call(call)

        # Need split to multiple parts according to the hardware vector width.
        part_cnt = math.ceil(dtype.lanes / hw_lanes)
        n = call.args[0]

        ret = []
        for i in range(part_cnt):
            cur_lanes = min(dtype.lanes - i * hw_lanes, hw_lanes)
            new_args = (S.clip(n - i * hw_lanes, 0, hw_lanes),)
            ret.append(tir.Call(dtype.with_lanes(cur_lanes), call.op, new_args, call.span))

        return tuple(ret)

    def _mutate_vload(self, call):
        dtype = call.dtype
        hw_lanes = self._get_hw_lanes(dtype)

        if dtype.lanes <= hw_lanes:
            return super().visit_call(call)

        # Need split to multiple parts according to the hardware vector width.
        part_cnt = math.ceil(dtype.lanes / hw_lanes)
        part_args = self._mutate_args(call.args, part_cnt)
        _, ptr, stride, _ = part_args[0]

        ret = []
        for i in range(part_cnt):
            cur_lanes = min(dtype.lanes - i * hw_lanes, hw_lanes)
            part_args[i][1] = _move_pointer_in_scalar_unit(ptr, i * hw_lanes * stride)
            ret.append(tir.Call(dtype.with_lanes(cur_lanes), call.op, part_args[i], call.span))

        return tuple(ret)

    def _create_new_call_if_needed(self, call, ret_args):
        if ret_args == list(call.args):
            return call
        return tir.Call(call.dtype, call.op, ret_args, call.span)

    def _mutate_vload_gather(self, call):
        ret_args = tuple(self.visit_expr(x) for x in call.args)
        dtype = call.dtype
        hw_lanes = self._get_hw_lanes(dtype)

        if all(not isinstance(x, tuple) for x in ret_args) and dtype.lanes <= hw_lanes:
            return self._create_new_call_if_needed(call, ret_args)

        # Need split to multiple parts according to the hardware vector width.
        part_cnt = math.ceil(dtype.lanes / hw_lanes)
        func_name, ptr, offsets, masks = ret_args
        offsets = (offsets,) if not isinstance(offsets, tuple) else offsets
        masks = (masks,) if not isinstance(masks, tuple) else masks
        masks = _get_new_bool(masks, part_cnt)

        ret = []
        for i in range(part_cnt):
            cur_lanes = min(dtype.lanes - i * hw_lanes, hw_lanes)

            if dtype.bits == 8:  # For 8-bit, each call consumes 1 ~ 2 offset variables.
                cur_offsets = offsets[i * 2 : i * 2 + 2]
            elif dtype.bits == 16:  # For 16-bit, each call consumes 1 offset variable.
                cur_offsets = (offsets[i],)
            else:  # For 32-bit, each call only consumes the low half part of the offset variable.
                cur_offsets = (offsets[i // 2] if i % 2 == 0 else _vshfl(offsets[i // 2], 8),)

            new_args = (func_name, ptr, *cur_offsets, masks[i])
            ret.append(tir.Call(dtype.with_lanes(cur_lanes), call.op, new_args, call.span))

        # The split result will only contain one expression when value is i8x32, offsets is u16x32,
        return ret[0] if len(ret) == 1 else tuple(ret)

    def _mutate_vcast(self, call):
        part_args = self._mutate_args(call.args)
        part_cnt_by_arg = len(part_args)
        to_dtype = call.dtype
        hw_lanes = self._get_hw_lanes(to_dtype)

        if part_cnt_by_arg == 1 and to_dtype.lanes <= hw_lanes:
            return self._create_new_call_if_needed(call, part_args[0])

        # Need split to multiple parts according to the hardware vector width.
        _, part, saturate, x = call.args
        no_zip = part == "all_with_no_zip"
        assert part.value[:3] == "all", 'Error in script API "S.cast".'
        from_bits, to_bits = x.dtype.bits, to_dtype.bits

        ret = []
        # 1. Cast to same bits, e.g., i8x55 -> u8x55, i32x11 -> fp32x11.
        if from_bits == to_bits:
            for i in range(part_cnt_by_arg):
                cur_lanes = min(to_dtype.lanes - i * hw_lanes, hw_lanes)
                ret_vdtype = to_dtype.with_lanes(cur_lanes)
                ret.append(tir.Call(ret_vdtype, call.op, part_args[i], call.span))

            return tuple(ret)

        # 2. Cast to wider bits, e.g., i8x55 -> i32x55, fp16x23 -> fp32x23.
        if from_bits < to_bits:
            part_strs = ("ll", "lh", "hl", "hh") if to_bits // from_bits == 4 else ("low", "high")
            part_strs = ("even", "odd") if no_zip else part_strs

            for part_arg in part_args:
                cur_x = part_arg[3]
                cur_x_lanes = cur_x.dtype.lanes
                cur_part_cnt_by_ret = math.ceil(cur_x_lanes / hw_lanes)

                for j in range(cur_part_cnt_by_ret):
                    new_args = ("vcast", part_strs[j], saturate, cur_x)
                    cur_lanes = min(cur_x_lanes - j * hw_lanes, hw_lanes)
                    ret_vdtype = to_dtype.with_lanes(cur_lanes)
                    ret.append(tir.Call(ret_vdtype, call.op, new_args, call.span))

            return tuple(ret)

        # 3. Cast to narrower bits without merge, e.g., i32x55 -> i8x55, fp32x23 -> fp16x23.
        max_input_cnt_per_part = from_bits // to_bits
        new_inputs = tuple(x[3] for x in part_args)
        part_cnt_by_ret = math.ceil(to_dtype.lanes / hw_lanes)

        for i in range(part_cnt_by_ret):
            cur_inputs = new_inputs[i * max_input_cnt_per_part : (i + 1) * max_input_cnt_per_part]
            cur_lanes = sum(x.dtype.lanes for x in cur_inputs)
            new_args = ("vcast", "all", saturate) + cur_inputs
            ret.append(tir.Call(to_dtype.with_lanes(cur_lanes), call.op, new_args, call.span))

        # The split result will only contain one expression when "to_dtype.lanes" <= "hw_lanes",
        # e.g., i32x32 -> i8x32, fp32x15 -> fp16x15.
        return ret[0] if len(ret) == 1 else tuple(ret)

    def _mutate_vzip(self, call):
        part_args = self._mutate_args(call.args)
        part_cnt_by_arg = len(part_args)
        dtype = call.dtype
        hw_lanes = self._get_hw_lanes(self._get_associated_dtype(call) if dtype.is_bool else dtype)

        if part_cnt_by_arg == 1 and dtype.lanes <= hw_lanes:
            return self._create_new_call_if_needed(call, part_args[0])

        # Need split to multiple parts according to the hardware vector width.
        part = call.args[3]

        # For "even" and "odd", only the feature Multiple Width Vector is supported.
        if part in ("even", "odd"):
            return tuple(tir.Call(x[1].dtype, call.op, x, call.span) for x in part_args)

        ret = []
        for part_arg in part_args:
            cur_x = part_arg[1]
            cur_part_cnt_by_ret = math.ceil(cur_x.dtype.lanes * 2 / hw_lanes)

            for j in range(cur_part_cnt_by_ret):
                new_args = list(part_arg)
                new_args[3] = ("low", "high")[j]
                ret.append(tir.Call(cur_x.dtype, call.op, new_args, call.span))

        # For "all", the feature Flexible Width Vector is also supported.
        if part == "all":
            return tuple(ret)

        # For "low" and "high", only the feature Multiple Width Vector is supported.
        return tuple(ret[:part_cnt_by_arg] if part == "low" else ret[part_cnt_by_arg:])

    def _mutate_vrevs(self, call):
        part_args = self._mutate_args(call.args)

        dtype = call.dtype
        hw_lanes = self._get_hw_lanes(self._get_associated_dtype(call) if dtype.is_bool else dtype)

        if len(part_args) == 1:
            return self._create_new_call_if_needed(call, part_args[0])

        hw_dtype = dtype.with_lanes(hw_lanes)
        return tuple([tir.Call(hw_dtype, call.op, x, call.span) for x in part_args[::-1]])

    def _mutate_vsldl(self, call):
        shift_imm = call.args[-1].value
        part_args = self._mutate_args(call.args)
        dtype = call.dtype
        hw_lanes = self._get_hw_lanes(dtype)
        num_multiples = len(part_args)

        if num_multiples == 1:
            return self._create_new_call_if_needed(call, part_args[0])

        shift_range_idx = shift_imm // hw_lanes
        new_shift_imm = shift_imm % hw_lanes

        args_concat = []
        for i in range(num_multiples):
            args_concat.append(part_args[i][1])
        for i in range(num_multiples):
            args_concat.append(part_args[i][2])

        if new_shift_imm == 0:
            return tuple(args_concat[shift_range_idx : len(args_concat) // 2 + shift_range_idx])

        ret = []
        for i in range(num_multiples):
            ret_dtype = part_args[i][1].dtype
            new_args = list(part_args[i])
            new_args[1] = args_concat[shift_range_idx + i]
            new_args[2] = args_concat[shift_range_idx + i + 1]
            new_args[3] = new_shift_imm
            ret.append(tir.Call(ret_dtype, call.op, new_args, call.span))
        return tuple(ret)

    def _mutate_vconcat(self, call):
        ret_args = list(self.visit_expr(x) for x in call.args)
        part = ret_args[-1]

        part_cnts, index_map = _get_part_cnt(ret_args, 1)
        if len(set(part_cnts)) != 1:
            # Here use 0 because the mask is always the last parameter.
            part_cnt = part_cnts[0]
            mask_indices = [index_map[i] for i, value in enumerate(part_cnts) if value != part_cnt]
            for i in mask_indices:
                masks = (ret_args[i],) if not isinstance(ret_args[i], tuple) else ret_args[i]
                msg = "Only boolean data type variable can be update during mutate args."
                assert all(mask.dtype.is_bool for mask in masks), msg
                new_bool = _get_new_bool(masks, part_cnt)
                ret_args[i] = new_bool[0] if len(new_bool) == 1 else new_bool
        else:
            part_cnt = part_cnts.pop()

        # Skip if: 2 inputs operand with hw dtype, and part is "lheo".
        if (
            len(call.args) == 4
            and all(not isinstance(x, tuple) for x in ret_args)
            and part != "all"
        ):
            return self._create_new_call_if_needed(call, ret_args)

        # Need split to multiple parts according to the hardware vector width.
        # Only support the feature Multiple Width Vector.
        operands = tuple(np.ravel(ret_args[1:-1]))
        if part == "all":
            return operands

        # From here, the count of operands must >= 2, it is guaranteed by parser.
        count = len(operands)
        inps = tuple(x if isinstance(x, tuple) else (x,) for x in ret_args[1:-1])
        if part in ("even", "odd"):
            if count % 2 == 1:
                useless = operands[-1]
                operands += (useless,)
                count += 1
            return tuple(S.vconcat(operands[2 * i : 2 * i + 2], part) for i in range(count // 2))

        # 1. The vector width is an even multiple, e.g., i32x16.
        half_part_cnt = part_cnt // 2
        if part_cnt % 2 == 0:
            ret = []
            for x in inps:
                ret += x[:half_part_cnt] if part == "low" else x[half_part_cnt:]
            return tuple(ret)

        # 2. The vector width is an odd multiple, e.g., i32x24.
        middle_part_idx = part_cnt // 2
        half_lanes = inps[0][0].dtype.lanes // 2
        ret = []
        # 2.1 part is "low":
        if part == "low":
            for i, x in enumerate(inps):
                if i % 2 == 0:
                    ret += list(x[:middle_part_idx])
                    # When index is the last one, concat with a useless element, here use the same.
                    next_one = x[middle_part_idx] if i == len(inps) - 1 else inps[i + 1][0]
                    ret.append(S.vconcat((x[middle_part_idx], next_one), "low"))
                else:
                    for j in range(middle_part_idx):
                        ret.append(S.vsldl(x[j], x[j + 1], half_lanes))
            return tuple(ret)

        # 2.2 part is "high":
        for i, x in enumerate(inps):
            if i % 2 == 0:
                for j in range(middle_part_idx, part_cnt - 1):
                    ret.append(S.vsldl(x[j], x[j + 1], half_lanes))
                next_one = x[-1] if i == len(inps) - 1 else inps[i + 1][middle_part_idx]
                ret.append(S.vconcat((x[-1], next_one), "high"))
            else:
                ret += list(x[middle_part_idx + 1 :])
        return tuple(ret)

    def _mutate_vsplit(self, call):
        _, x, idx = call.args
        return self.visit_expr(x)[idx.value]

    def _mutate_reassign(self, call):
        part_args = self._mutate_args(call.args)
        part_cnt = len(part_args)

        if part_cnt == 1:
            return self._create_new_call_if_needed(call, part_args[0])

        # Need split to multiple parts according to the hardware vector width.
        ret = []
        for part_arg in part_args:
            var, value = part_arg
            if is_builtin(var, "vload"):
                _, ptr, stride, mask = var.args
                ret.append(_vstore(value, ptr, stride, mask))
            else:
                ret.append(tir.Call("void", call.op, part_arg, call.span))
        return tuple(ret)

    def _mutate_pointer(self, call):
        part_args = self._mutate_args(call.args)
        part_cnt = len(part_args)

        if part_cnt == 1:
            return self._create_new_call_if_needed(call, part_args[0])

        # Indicate this pointer is created by getting address of a virtual vector variable. The
        # virtual vector variable is changed to an array, so here need to change the base of the
        # pointer to the base address of the new array.
        part_args[0][2] = self._var2buffer[call.args[2]].data
        return tir.Call(call.dtype, call.op, part_args[0], call.span)

    def _mutate_vrpadd(self, call):
        part_args = self._mutate_args(call.args)
        dtype = call.dtype
        part_cnt = len(part_args)
        hw_lanes = _ceil_to_multiple_of_8(dtype.lanes / part_cnt)

        if part_cnt == 1:
            return self._create_new_call_if_needed(call, part_args[0])

        hw_dtype = dtype.with_lanes(hw_lanes)
        ret0 = tir.Call(hw_dtype, call.op, part_args[0], call.span)
        for i in range(1, part_cnt):
            ret0 += tir.Call(hw_dtype, call.op, part_args[i], call.span)
        return tuple([ret0] + [tir.const(0, hw_dtype)] * (part_cnt - 1))

    def _mutate_horizontal_op(self, call):
        ret_args = tuple(self.visit_expr(x) for x in call.args)

        if all(not isinstance(x, tuple) for x in ret_args):
            return self._create_new_call_if_needed(call, ret_args)

        # Need split to multiple parts according to the hardware vector width.
        # Only support the feature Multiple Width Vector.
        func_name, x, y = ret_args
        part_cnt = len(x)
        dtype = call.dtype
        hw_lanes = _ceil_to_multiple_of_8(dtype.lanes / part_cnt)

        ret = []
        new_inputs = x + y
        for i in range(part_cnt):
            new_args = (func_name, new_inputs[i * 2], new_inputs[i * 2 + 1])
            ret.append(tir.Call(dtype.with_lanes(hw_lanes), call.op, new_args, call.span))

        return tuple(ret)

    def _mutate_vector_set_get_element(self, call):
        part_args = self._mutate_args(call.args)

        if len(part_args) == 1:
            return self._create_new_call_if_needed(call, part_args[0])

        dtype = call.args[0].dtype
        hw_lanes = self._get_hw_lanes(dtype)
        idx = int(call.args[1])

        selected_vector_idx = idx // hw_lanes
        new_idx = idx % hw_lanes

        new_args = list(part_args[selected_vector_idx])
        new_args[1] = new_idx
        return tir.Call(call.dtype, call.op, new_args, call.span)

    def _mutate_default(self, call):
        part_args = self._mutate_args(call.args)
        part_cnt = len(part_args)

        if part_cnt == 1:
            return self._create_new_call_if_needed(call, part_args[0])

        # Need split to multiple parts according to the hardware vector width.
        dtype = call.dtype
        hw_lanes = _ceil_to_multiple_of_8(dtype.lanes / part_cnt)

        ret = []
        for i in range(part_cnt):
            cur_lanes = min(dtype.lanes - i * hw_lanes, hw_lanes)
            ret.append(tir.Call(dtype.with_lanes(cur_lanes), call.op, part_args[i], call.span))

        return tuple(ret)

    def _create_new_vars(self, var, values):
        var_constructor = tir.SizeVar if isinstance(var, tir.SizeVar) else tir.Var

        if var.dtype.is_bool_vector:
            # For the mask variables, the split variable is changed to multiple new variables,
            # because mask array will make the analysis "get_mask_associated_dtype" more difficult.
            return tuple(var_constructor(f"{var.name}_{i}", x.dtype) for i, x in enumerate(values))

        # For the variables other than mask, the split variable is changed to an array, so we can
        # support to get the address of the virtual vector variable.
        buf_dtype = hw_native_vdtype(values[0].dtype)
        buf_var = var_constructor(var.name, ir.PointerType(ir.PrimType(buf_dtype), "local"))
        buf = tir.decl_buffer((len(values),), buf_dtype, f"{var.name}_buf", data=buf_var)
        self._var2buffer[var] = buf
        return tuple(_vload(buf.addr_of(i), x.dtype.lanes) for i, x in enumerate(values))

    def visit_let_stmt(self, let_stmt):
        new_value = self.visit_expr(let_stmt.value)

        var = let_stmt.var
        if isinstance(new_value, tuple):
            self._var_substitute_map[var] = self._create_new_vars(var, new_value)

        new_body = self.visit_stmt(let_stmt.body)

        if new_value == let_stmt.value and new_body == let_stmt.body:
            return let_stmt
        if not isinstance(new_value, tuple):
            return tir.LetStmt(var, new_value, new_body, let_stmt.span)

        # Indicate this let statement is split to multiple parts.
        if var.dtype.is_bool_vector:
            # For the mask variables, the split variable is changed to multiple new variables, each
            # part corresponding to a new let statement, nest them in reverse order.
            for var, value in reversed(list(zip(self._var_substitute_map[var], new_value))):
                new_body = tir.LetStmt(var, value, new_body, let_stmt.span)
            return new_body

        # For the variables other than mask, the split variable is changed to an array, each part
        # corresponding to a new buffer store statement, and the original let statement is changed
        # to an allocation statement.
        buf = self._var2buffer[var]
        ret = tuple(tir.Evaluate(_vstore(x, buf.addr_of(i))) for i, x in enumerate(new_value))
        ret = tir.SeqStmt(ret + (new_body,))
        return tir.Allocate(
            buf.data,
            buf.dtype,
            buf.shape,
            tir.const(1, "bool"),
            tir.DeclBuffer(buf, ret),
            {"need_revert_load_store": True},
        )

    def _mutate_vstore(self, call):
        part_args = self._mutate_args(call.args)
        part_cnt = len(part_args)

        if part_cnt == 1:
            return self._create_new_call_if_needed(call, part_args[0])

        # Need split to multiple parts according to the hardware vector width.
        dtype = call.args[1].dtype
        hw_lanes = _ceil_to_multiple_of_8(dtype.lanes / part_cnt)
        _, _, ptr, stride, _ = part_args[0]

        ret = []
        for i in range(part_cnt):
            part_args[i][2] = _move_pointer_in_scalar_unit(ptr, i * hw_lanes * stride)
            ret.append(tir.Call("void", call.op, part_args[i], call.span))

        return tuple(ret)

    def _mutate_vstore_scatter(self, call):
        ret_args = tuple(self.visit_expr(x) for x in call.args)
        dtype = call.args[1].dtype
        hw_lanes = self._get_hw_lanes(dtype)

        if all(not isinstance(x, tuple) for x in ret_args) and dtype.lanes <= hw_lanes:
            return self._create_new_call_if_needed(call, ret_args)

        # Need split to multiple parts according to the hardware vector width.
        part_cnt = math.ceil(dtype.lanes / hw_lanes)
        func_name, values, ptr, offsets, masks = ret_args
        values = (values,) if not isinstance(values, tuple) else values
        offsets = (offsets,) if not isinstance(offsets, tuple) else offsets
        masks = (masks,) if not isinstance(masks, tuple) else masks
        masks = _get_new_bool(masks, part_cnt)

        ret = []
        for i in range(part_cnt):
            if dtype.bits == 8:  # For 8-bit, each call consumes 1 ~ 2 offset variables.
                cur_offsets = offsets[i * 2 : i * 2 + 2]
            elif dtype.bits == 16:  # For 16-bit, each call consumes 1 offset variable.
                cur_offsets = (offsets[i],)
            else:  # For 32-bit, each call only consumes the low half part of the offset variable.
                cur_offsets = (offsets[i // 2] if i % 2 == 0 else _vshfl(offsets[i // 2], 8),)

            new_args = (func_name, values[i], ptr, *cur_offsets, masks[i])
            ret.append(tir.Call("void", call.op, new_args, call.span))

        # The split result will only contain one expression when value is i8x32, offsets is u16x32,
        return ret[0] if len(ret) == 1 else tuple(ret)

    def visit_evaluate(self, evaluate):
        new_value = self.visit_expr(evaluate.value)

        if new_value == evaluate.value:
            return evaluate
        if not isinstance(new_value, tuple):
            return tir.Evaluate(new_value, evaluate.span)

        # Indicate it's split to multiple parts, combine them into one statement by "tir.SeqStmt".
        return tir.SeqStmt([tir.Evaluate(x, evaluate.span) for x in new_value], evaluate.span)

    def _mutate_vall_vany(self, call):
        part_args = self._mutate_args(call.args)
        part_cnt = len(part_args)

        if part_cnt == 1:
            return self._create_new_call_if_needed(call, part_args[0])

        tir_func = tir.all if call.args[0].value == "vall" else tir.any
        return tir_func(*[tir.Call(call.dtype, call.op, x, call.span) for x in part_args])

    def _mutate_vreplic(self, call):
        part_args = self._mutate_args(call.args)
        part_cnt = len(part_args)

        if part_cnt == 1:
            return self._create_new_call_if_needed(call, part_args[0])

        dtype = call.dtype
        hw_lanes = _ceil_to_multiple_of_8(dtype.lanes / part_cnt)
        index = int(call.args[2])
        part_idx = index // hw_lanes
        index_offset = index - part_idx * hw_lanes

        part_args[part_idx][2] = index_offset
        ret = [tir.Call(dtype.with_lanes(hw_lanes), call.op, part_args[part_idx], call.span)]
        return tuple(ret * part_cnt)

    def _mutate_vbcast(self, call):
        dtype = call.dtype
        if dtype.is_bool:
            part_args = self._mutate_args(call.args)
            part_cnt = len(part_args)
            hw_lanes = _ceil_to_multiple_of_8(dtype.lanes / part_cnt)
        else:
            hw_lanes = self._get_hw_lanes(dtype)
            part_cnt = math.ceil(dtype.lanes / hw_lanes)
            part_args = self._mutate_args(call.args, part_cnt)

        if part_cnt == 1:
            return self._create_new_call_if_needed(call, part_args[0])

        ret = []
        for i in range(part_cnt):
            cur_lanes = min(dtype.lanes - i * hw_lanes, hw_lanes)
            ret.append(tir.Call(dtype.with_lanes(cur_lanes), call.op, part_args[i], call.span))

        return tuple(ret)

    def _mutate_global_var(self, call):
        new_args = []
        for x in call.args:
            new_arg = self.visit_expr(x)
            new_arg = list(new_arg) if isinstance(new_arg, tuple) else [new_arg]
            new_args += new_arg

        if new_args == list(call.args):
            return call

        return tir.Call(call.dtype, call.op, new_args, call.span)

    def _mutate_vector_literal(self, call):
        dtype = call.dtype
        real_dtype = self._get_associated_dtype(call) if dtype.is_bool else dtype
        hw_lanes = self._get_hw_lanes(real_dtype)

        if dtype.lanes <= hw_lanes:
            return super().visit_call(call)

        # Need split to multiple parts according to the hardware vector width.
        part_cnt = math.ceil(dtype.lanes / hw_lanes)
        part_args = self._mutate_args(call.args)[0]

        ret = []
        for i in range(part_cnt):
            cur_lanes = min(dtype.lanes - i * hw_lanes, hw_lanes)
            args = part_args[i * hw_lanes : i * hw_lanes + cur_lanes]
            ret.append(tir.Call(dtype.with_lanes(cur_lanes), call.op, args, call.span))

        return tuple(ret)

    def visit_call(self, call):
        if call.op == ir.Op.get("tir.const_pred"):
            return self._mutate_const_pred(call)
        if call.op == ir.Op.get("tir.low_true_pred"):
            return self._mutate_low_true_pred(call)
        if call.op == ir.Op.get("tir.reassign"):
            return self._mutate_reassign(call)
        if call.op == ir.Op.get("tir.pointer"):
            return self._mutate_pointer(call)
        if call.op == ir.Op.get("tir.reinterpret"):
            return self._mutate_default(call)
        if call.op == ir.Op.get("tir.vector_literal"):
            return self._mutate_vector_literal(call)
        if isinstance(call.op, ir.GlobalVar):
            return self._mutate_global_var(call)
        if call.op in [ir.Op.get("tir.vector_set_element"), ir.Op.get("tir.vector_get_element")]:
            return self._mutate_vector_set_get_element(call)

        if call.op != ir.Op.get("tir.call_extern"):
            return super().visit_call(call)

        func_name = call.args[0].value
        if func_name == "vload":
            return self._mutate_vload(call)
        if func_name == "vstore":
            return self._mutate_vstore(call)
        if func_name == "vload_gather":
            return self._mutate_vload_gather(call)
        if func_name == "vstore_scatter":
            return self._mutate_vstore_scatter(call)
        if func_name == "vcast":
            return self._mutate_vcast(call)
        if func_name == "vzip":
            return self._mutate_vzip(call)
        if func_name == "vrevs":
            return self._mutate_vrevs(call)
        if func_name == "__vsldl":
            return self._mutate_vsldl(call)
        if func_name == "vconcat":
            return self._mutate_vconcat(call)
        if func_name == "vsplit":
            return self._mutate_vsplit(call)
        if func_name == "__vrpadd":
            return self._mutate_vrpadd(call)
        if func_name in ("__vaddh", "__vsubh", "__vmaxh", "__vminh"):
            return self._mutate_horizontal_op(call)
        if func_name in ("vall", "vany"):
            return self._mutate_vall_vany(call)
        if func_name == "__vreplic":
            return self._mutate_vreplic(call)
        if func_name == "__vbcast":
            return self._mutate_vbcast(call)
        if func_name in _default_funcs:
            return self._mutate_default(call)

        return super().visit_call(call)


def _split_param(params, cps_info, mask_asso_dtype):
    new_params = []
    var_substitute_map = {}
    pad_param_ids = []
    for param in params:
        if param.dtype != "handle":
            dtype = param.dtype
            asso_bits = mask_asso_dtype[param].bits if dtype.is_bool else dtype.bits
            hw_lanes = cps_info.vector_width // asso_bits
            if dtype.lanes > hw_lanes:
                part_cnt = math.ceil(dtype.lanes / hw_lanes)
                var_constructor = tir.SizeVar if isinstance(param, tir.SizeVar) else tir.Var
                for i in range(part_cnt):
                    cur_lanes = min(dtype.lanes - i * hw_lanes, hw_lanes)
                    cur_dtype = dtype.with_lanes(cur_lanes)
                    new_params.append(var_constructor(f"{param.name}_{i}", cur_dtype))
                    if cur_lanes < hw_lanes:
                        pad_param_ids.append(len(new_params) - 1)
                var_substitute_map[param] = tuple(new_params[-part_cnt:])
            else:
                new_params.append(param)
        else:
            new_params.append(param)
    return new_params, var_substitute_map, pad_param_ids


@ir.transform.module_pass(opt_level=0)
class AlignVectorWidthBySplit:
    """Align the width of all wider vector nodes with the hardware vector width by split.

    Precondition
      1. The different indirect usages of the same mask node should have the same data type, need to
         be checked by "get_mask_associated_dtype".
    """

    def __init__(self, cps_info):
        self._cps_info = cps_info

    def transform_function(self, func, mask_asso_dtype):
        """Traverse the given PrimFunc, transform it and return the result."""
        new_params, var_substitute_map, pad_param_ids = _split_param(
            func.params, self._cps_info, mask_asso_dtype
        )
        new_body = _Mutator(self._cps_info, mask_asso_dtype, var_substitute_map).visit(func.body)
        if new_params == list(func.params):
            return func.with_body(new_body, span=func.span)
        new_func = tir.PrimFunc(
            new_params,
            new_body,
            func.ret_type,
            func.buffer_map,
            func.attrs,
            func.span,
        )
        if len(pad_param_ids) != 0:
            new_func = new_func.with_attr("need_pad_param_ids", pad_param_ids)
        return new_func

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """try transform module"""
        var2mask_asso_dtype = get_mask_associated_dtype(mod)
        for var, func in mod.functions.items():
            new_func = self.transform_function(func, var2mask_asso_dtype[var])
            mod.update_func(var, new_func)
        return mod
