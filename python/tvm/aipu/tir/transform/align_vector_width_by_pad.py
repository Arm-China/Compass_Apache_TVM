# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Align the width of all narrower vector nodes with the hardware vector width by pad."""
from tvm import tir, ir, DataType
from ..analysis import ensure_well_formed, get_mask_associated_dtype
from ... import script as S


_default_funcs = ("vload", "__vbcast", "vclt", "vcgt", "vcle", "vcge", "vceq", "vcneq", "vxor")
_default_funcs += ("vadd", "vsub", "vmul", "__vdiv", "vexp", "vtanh", "vlog", "__vrint", "vabs")
_default_funcs += ("__vsel", "vpow", "__vfma", "vsin", "vcos", "vrsqrt", "vsqrt", "vfloor", "vceil")
_default_funcs += ("vsl", "vsr", "vror", "__vcls", "__vclz", "__vmod", "vand", "vor", "vinv")
_default_funcs += ("__vbrevs", "__vpcnt", "__vmax", "__vmin", "__vclass")


_ERR_IN_SPLIT_MSG = 'Error in "AlignVectorWidthBySplit".'


class _Mutator(tir.StmtExprMutator):
    def __init__(self, aipu_info, mask2associated_dtype, var_substitute_map):
        super().__init__()
        self._aipu_info = aipu_info
        self._mask2associated_dtype = mask2associated_dtype
        self._var_substitute_map = var_substitute_map

    def visit_var(self, var):
        return self._var_substitute_map.get(var, var)

    def _get_associated_dtype(self, call):
        dtype = DataType(call.dtype)
        assert dtype.is_bool, "Only can get associated dtype through boolean data type."
        if call in self._mask2associated_dtype:
            return self._mask2associated_dtype[call]
        return dtype.with_bits(self._aipu_info.vector_width // dtype.lanes)

    def _get_hw_lanes(self, dtype):
        assert not dtype.is_bool, "Can't get hardware lanes through boolean data type."
        return self._aipu_info.vector_width // dtype.bits

    def _mutate_const_pred(self, call):
        ret = super().visit_call(call)

        dtype = DataType(ret.dtype)
        hw_lanes = self._get_hw_lanes(self._get_associated_dtype(call))
        if dtype.lanes == hw_lanes:
            return ret

        assert dtype.lanes < hw_lanes, _ERR_IN_SPLIT_MSG
        # AIPU's vector register is 256-bit, and can't be split into small parts for use, so if we
        # want to load a int8x8 data, the predicate must be set to 0x0000_00FF. For those data whose
        # total bits < 256, need to complement "false" to the high position.
        new_args = list(ret.args) + [False] * (hw_lanes - dtype.lanes)
        return tir.Call(dtype.with_lanes(hw_lanes), ret.op, new_args, ret.span)

    def _mutate_low_true_pred(self, call):
        ret = super().visit_call(call)

        dtype = DataType(ret.dtype)
        hw_lanes = self._get_hw_lanes(self._get_associated_dtype(call))
        if dtype.lanes == hw_lanes:
            return ret

        assert dtype.lanes < hw_lanes, _ERR_IN_SPLIT_MSG
        return tir.Call(dtype.with_lanes(hw_lanes), ret.op, ret.args, ret.span)

    def _get_new_lanes(self, old_args, new_args):
        ret = None
        for old_arg, new_arg in zip(old_args, new_args):
            old_lanes = DataType(old_arg.dtype).lanes
            new_lanes = DataType(new_arg.dtype).lanes
            if old_lanes != new_lanes:
                if ret is None:
                    ret = new_lanes
                assert new_lanes == ret, "The pad lanes of all vector arguments must be same."
        return ret

    def _mutate_vcast(self, call):
        ret = super().visit_call(call)

        from_new_lanes = self._get_new_lanes(call.args, ret.args)
        to_dtype = DataType(ret.dtype)
        to_hw_lanes = self._get_hw_lanes(to_dtype)
        if from_new_lanes is None and to_dtype.lanes == to_hw_lanes:
            return ret

        assert to_dtype.lanes <= to_hw_lanes, _ERR_IN_SPLIT_MSG
        new_args = list(ret.args)
        _, part, _, *inputs = new_args

        # 1. Cast to narrower bits with merge, e.g., (fp32x8, fp32x3) -> fp16x11,
        #    (i32x8, i32x8, i32x8) -> i8x24, (i16x16, i16x11) -> i8x27
        if len(inputs) > 1:
            assert part == "all" and to_dtype.lanes < to_hw_lanes, _ERR_IN_SPLIT_MSG
            assert sum(DataType(x.dtype).lanes for x in inputs) <= to_hw_lanes, _ERR_IN_SPLIT_MSG
            return tir.Call(to_dtype.with_lanes(to_hw_lanes), ret.op, new_args, ret.span)

        from_bits, to_bits = DataType(inputs[0].dtype).bits, to_dtype.bits
        msg = 'Error in script API "S.cast" or "AlignVectorWidthBySplit".'

        if from_bits == to_bits:
            # 2. Cast to same bits, e.g., i8x16 -> u8x16, i32x4 -> fp32x4.
            assert part == "all", msg
            assert from_new_lanes is not None and to_dtype.lanes < to_hw_lanes, msg
        elif from_bits < to_bits:
            # 3. Cast to wider bits, e.g., i8x8 -> i16x8, fp16x8 -> fp32x8.
            assert from_new_lanes is not None, msg
            if part == "all":  # e.g., i8x8 -> i32x8, fp16x8 -> i32x8.
                new_args[1] = "ll" if to_bits // from_bits == 4 else "low"
        else:
            # 4. Cast to narrower bits without merge, e.g., i32x8 -> i8x8, fp32x8 -> fp16x8.
            assert part == "all" and to_dtype.lanes < to_hw_lanes, msg

        return tir.Call(to_dtype.with_lanes(to_hw_lanes), ret.op, new_args, ret.span)

    def _mutate_vzip(self, call):
        ret = super().visit_call(call)

        dtype = DataType(ret.dtype)
        new_lanes = self._get_new_lanes(call.args, ret.args)
        if new_lanes is None:
            return ret

        assert dtype.lanes <= new_lanes, _ERR_IN_SPLIT_MSG
        new_args = list(ret.args)
        new_args[3] = "low" if new_args[3] == "all" else new_args[3]
        return tir.Call(dtype.with_lanes(new_lanes), ret.op, new_args, ret.span)

    def _mutate_vload_gather(self, call):
        ret = super().visit_call(call)

        dtype = DataType(ret.dtype)
        hw_lanes = self._get_hw_lanes(dtype)
        if dtype.lanes == hw_lanes:
            return ret

        assert dtype.lanes <= hw_lanes, _ERR_IN_SPLIT_MSG
        new_args = list(ret.args)
        if dtype.bits == 8 and dtype.lanes <= 16:  # e.g., i8x12, i8x16.
            new_args.insert(3, S.u16x16(0))  # Compass OpenCL need two "offsets" arguments.
        return tir.Call(dtype.with_lanes(hw_lanes), ret.op, new_args, ret.span)

    def _mutate_vstore_scatter(self, call):
        ret = super().visit_call(call)

        dtype = DataType(call.args[1].dtype)
        hw_lanes = self._get_hw_lanes(dtype)
        if dtype.lanes == hw_lanes:
            return ret

        assert dtype.lanes <= hw_lanes, _ERR_IN_SPLIT_MSG
        new_args = list(ret.args)
        if dtype.bits == 8 and dtype.lanes <= 16:  # e.g., i8x12, i8x16.
            new_args.insert(4, S.u16x16(0))  # Compass OpenCL need two "offsets" arguments.
        return tir.Call("void", ret.op, new_args, ret.span)

    def _mutate_default(self, call):
        ret = super().visit_call(call)

        dtype = DataType(ret.dtype)
        new_lanes = self._get_new_lanes(call.args, ret.args)
        if new_lanes is None or dtype.lanes == new_lanes:
            return ret

        assert dtype.lanes < new_lanes, _ERR_IN_SPLIT_MSG
        return tir.Call(dtype.with_lanes(new_lanes), ret.op, ret.args, ret.span)

    def _mutate_vall(self, call):
        ret = super().visit_call(call)

        old_lanes = DataType(call.args[1].dtype).lanes
        new_lanes = DataType(ret.args[1].dtype).lanes
        if old_lanes == new_lanes:
            return ret

        assert old_lanes < new_lanes, _ERR_IN_SPLIT_MSG

        # Indicate the complemented part of "vall" must be true for the result to remain unffected.
        new_args = list(ret.args)
        new_args[1] = S.vor(ret.args[1], S.const_mask(f"{old_lanes}F{new_lanes - old_lanes}T"))

        return tir.Call(ret.dtype, ret.op, new_args, ret.span)

    def _mutate_vector_literal(self, call):
        ret = super().visit_call(call)

        dtype = DataType(ret.dtype)
        hw_lanes = self._get_hw_lanes(dtype)
        if dtype.lanes == hw_lanes:
            return ret

        assert dtype.lanes < hw_lanes, _ERR_IN_SPLIT_MSG
        new_args = list(ret.args) + [0] * (hw_lanes - dtype.lanes)
        return tir.Call(dtype.with_lanes(hw_lanes), ret.op, new_args, ret.span)

    def visit_call(self, call):
        if call.op == ir.Op.get("tir.const_pred"):
            return self._mutate_const_pred(call)
        if call.op == ir.Op.get("tir.low_true_pred"):
            return self._mutate_low_true_pred(call)
        if call.op == ir.Op.get("tir.vector_literal"):
            return self._mutate_vector_literal(call)

        if call.op != ir.Op.get("tir.call_extern"):
            return super().visit_call(call)

        func_name = call.args[0].value
        if func_name == "vcast":
            return self._mutate_vcast(call)
        if func_name == "vzip":
            return self._mutate_vzip(call)
        if func_name == "vload_gather":
            return self._mutate_vload_gather(call)
        if func_name == "vstore_scatter":
            return self._mutate_vstore_scatter(call)
        if func_name == "vall":
            return self._mutate_vall(call)
        if func_name in _default_funcs:
            return self._mutate_default(call)

        return super().visit_call(call)

    def visit_let_stmt(self, let_stmt):
        new_value = self.visit_expr(let_stmt.value)

        new_var = let_stmt.var
        if new_value.dtype != let_stmt.value.dtype:
            var = let_stmt.var
            var_constructor = tir.SizeVar if isinstance(var, tir.SizeVar) else tir.Var
            new_var = var_constructor(var.name, new_value.dtype, var.span)
            self._var_substitute_map[var] = new_var

        new_body = self.visit_stmt(let_stmt.body)

        if new_value == let_stmt.value and new_body == let_stmt.body:
            return let_stmt
        return tir.LetStmt(new_var, new_value, new_body, let_stmt.span)


def _pad_params(func, aipu_info, mask_asso_dtype):
    new_params = list(func.params)
    var_substitute_map = {}
    need_pad_param_ids = func.attrs.get("need_pad_param_ids", [])
    for i, param in enumerate(func.params):
        if param.dtype != "handle":
            dtype = DataType(param.dtype)
            asso_bits = mask_asso_dtype[param].bits if dtype.is_bool else dtype.bits
            hw_lanes = aipu_info.vector_width // asso_bits
            assert dtype.lanes <= hw_lanes, _ERR_IN_SPLIT_MSG
            var_constructor = tir.SizeVar if isinstance(param, tir.SizeVar) else tir.Var
            if 1 < dtype.lanes < hw_lanes or i in need_pad_param_ids:
                new_params[i] = var_constructor(param.name, dtype.with_lanes(hw_lanes))
                var_substitute_map[param] = new_params[i]
    return new_params, var_substitute_map


@ir.transform.module_pass(opt_level=0)
class AlignVectorWidthByPad:
    """Align the width of all narrower vector nodes with the hardware vector width by pad.

    Precondition
      1. The different indirect usages of the same mask node should have the same data type, need to
         be checked by "get_mask_associated_dtype".
    """

    def __init__(self, aipu_info):
        self._aipu_info = aipu_info

    def transform_function(self, func, mask_asso_dtype):
        """Traverse the given PrimFunc, transform it and return the result."""
        new_params, var_substitute_map = _pad_params(func, self._aipu_info, mask_asso_dtype)
        new_body = _Mutator(self._aipu_info, mask_asso_dtype, var_substitute_map).visit(func.body)
        new_func = tir.PrimFunc(
            new_params,
            new_body,
            func.ret_type,
            func.buffer_map,
            func.attrs,
            func.span,
        )
        if "need_pad_param_ids" in func.attrs.keys():
            new_func = new_func.without_attr("need_pad_param_ids")
        ensure_well_formed(new_func)
        return new_func

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        var2mask_asso_dtype = get_mask_associated_dtype(mod)
        for var, func in mod.functions.items():
            new_func = self.transform_function(func, var2mask_asso_dtype[var])
            mod.update_func(var, new_func)
        return mod
