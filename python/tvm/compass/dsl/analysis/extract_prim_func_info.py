# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Extract the primitive function information through analyze the TIR IRModule."""
from tvm import tir, ir
from ..utils import is_pointer


class ParamInfo:
    """The class record the information of a primitive function parameter."""

    def __init__(self, name, dtype, category, tensor_idx=None):
        self.name = name
        self.dtype = dtype
        self.category = category
        self.tensor_idx = tensor_idx

    @property
    def is_input_tensor(self):
        return self.category == "input_tensor"

    @property
    def is_const_tensor(self):
        return self.category == "const_tensor"

    @property
    def is_output_tensor(self):
        return self.category == "output_tensor"

    @property
    def is_descriptor(self):
        return self.category == "descriptor"

    @property
    def is_attr(self):
        return self.category == "attr"

    @property
    def is_tensor(self):
        return self.is_input_tensor or self.is_const_tensor or self.is_output_tensor

    def __repr__(self):
        return f"{self.name}:{self.category},{self.dtype}"


class PrimFuncInfo:
    """The class record the information of a primitive function."""

    def __init__(self, name, param_cnt):
        self.name = name
        self.param_infos = [None] * param_cnt


class _ChainItem:
    def __init__(self, blk_id, ptr_var=None, param_idx=None):
        self.blk_id = blk_id
        self.ptr_var = ptr_var
        self.param_idx = param_idx

    @property
    def is_endpoint(self):
        return self.param_idx is not None and self.ptr_var is None


class _Analyser(tir.StmtExprVisitor):
    def __init__(self, func, ir_mod):
        super().__init__()
        self.written_pointer_param_indices = set()
        self.desc_pointer_param_indices = set()
        self._ir_mod = ir_mod
        self._cur_blk_id = 0
        self._alias2ptrs = {}
        self._call2ret_ptr_associated_args = {}
        self._ret_ptr_alias_param_indices = set()
        self._func_name = func.attrs["global_symbol"]
        self._func_params = func.params

        for i, x in enumerate(func.params):
            if x.dtype == "handle":
                self._alias2ptrs[x] = {0: {_ChainItem(0, param_idx=i)}}

        self.visit(func.body)

    def _get_origin_vars_of_ptr(self, ptr_call):
        assert is_pointer(ptr_call)

        # For "tir.pointer", the argument "base" can be one of below types.
        # 1. Var(type_annotation=PointerType): A usual pointer.
        # 2. Var(type_annotation=PrimType): A pointer created by getting address of a variable.
        # 3. BufferLoad: Same as 2, the var node is replaced by pass "ReassignVarBy0DimBuffer".
        # 4. Call(op="tir.pointer"): A pointer created by type conversion interface "as_ptr".
        # 5. Call(op="tir.reinterpret"): A pointer created by type reinterpret interface.
        # 6. Call(op=GlobalVar): A pointer created by a sub function return value.
        base = ptr_call.args[2]

        # For the situation 2 and 3, because the pointers that they represent must have nothing to
        # do with the written pointer parameters, so just return the "base" like the situation 1 is
        # enough.
        if isinstance(base, (tir.Var, tir.BufferLoad)):
            return {base}

        # For the situation 4, just need to do recursion.
        if is_pointer(base):
            return self._get_origin_vars_of_ptr(base)

        # For the situation 5, return the "base" of reinterpret interface.
        if isinstance(base, tir.Call) and base.op == ir.Op.get("tir.reinterpret"):
            return {base.args[0]}

        # For the situation 6, the result depend on all associated arguments of the sub function.
        ret = set()
        for ptr_arg in self._call2ret_ptr_associated_args[base]:
            ret |= self._get_origin_vars_of_ptr(ptr_arg)
        return ret

    def _find_param_indices_by_ptr_var(self, ptr_var, blk_id):
        if ptr_var not in self._alias2ptrs:
            return set()

        ret = set()
        for chain_item in self._alias2ptrs[ptr_var][blk_id]:
            if chain_item.is_endpoint:
                return {chain_item.param_idx}
            ret |= self._find_param_indices_by_ptr_var(chain_item.ptr_var, chain_item.blk_id)

        return ret

    def _find_param_indices(self, ptr_or_ptr_var):
        if is_pointer(ptr_or_ptr_var):
            ptr_vars = self._get_origin_vars_of_ptr(ptr_or_ptr_var)
        else:
            ptr_vars = {ptr_or_ptr_var}

        ret = set()
        for ptr_var in ptr_vars:
            ret |= self._find_param_indices_by_ptr_var(ptr_var, self._cur_blk_id)
        return ret

    def visit_buffer_store(self, buf_store):
        super().visit_buffer_store(buf_store)
        self.written_pointer_param_indices |= self._find_param_indices(buf_store.buffer.data)

    def visit_call(self, call):
        super().visit_call(call)

        if call.op == ir.Op.get("tir.reassign"):
            var, value = call.args
            if not is_pointer(value):
                return

            origin_vars = self._get_origin_vars_of_ptr(value)
            chain_items = {_ChainItem(self._cur_blk_id, x) for x in origin_vars if x != var}
            if var in origin_vars:  # Maybe a parameter, so need keep the endpoint chain items.
                chain_items |= {x for x in self._alias2ptrs[var][self._cur_blk_id] if x.is_endpoint}
            self._alias2ptrs[var][self._cur_blk_id] = chain_items
            return

        if isinstance(call.op, ir.GlobalVar):
            callee = self._ir_mod[call.op.name_hint]
            sub_func_analyser = _Analyser(callee, self._ir_mod)
            for i in sub_func_analyser.written_pointer_param_indices:
                self.written_pointer_param_indices |= self._find_param_indices(call.args[i])
            for i in sub_func_analyser.desc_pointer_param_indices:
                self.desc_pointer_param_indices |= self._find_param_indices(call.args[i])

            if isinstance(callee.ret_type, ir.PointerType):
                indices = sub_func_analyser._ret_ptr_alias_param_indices
                self._call2ret_ptr_associated_args[call] = {call.args[i] for i in indices}
            return

        if call.op == ir.Op.get("tir.ret"):
            if len(call.args) != 0 and is_pointer(call.args[0]):
                ret_ptr_alias_param_indices = self._find_param_indices(call.args[0])

                # No any parameter is associated with the returned pointer, means it's alloced in
                # the function.
                msg = f'The function "{self._func_name}" return a stack pointer. Please check it '
                msg += "carefully."
                assert len(ret_ptr_alias_param_indices) != 0, msg
                self._ret_ptr_alias_param_indices |= ret_ptr_alias_param_indices
            return

        if call.op == ir.Op.get("tir.call_extern"):
            ptr = None
            func_name = call.args[0].value
            if func_name in ("AsyncDmaDirect", "DmaDirect", "DmaUpsample"):
                if call.args[1].args[1].value.startswith("global"):
                    ptr = call.args[1]
            elif func_name in ("vstore", "vstore_scatter", "DMA_Transpose2D"):
                ptr = call.args[2]

            if ptr is not None:
                self.written_pointer_param_indices |= self._find_param_indices(ptr)

            if func_name in ("AIFF", "ASYNC_AIFF"):
                indices = set()
                for i in range(1, 4):
                    indices |= self._find_param_indices(call.args[i])

                for idx in indices:
                    dtype = self._func_params[idx].type_annotation.element_type.dtype
                    msg = f'The descriptor parameter "{self._func_params[idx].name}" of function '
                    msg += f'"{self._func_name}" expect dtype "uint32/int32", but got: "{dtype}".'
                    assert dtype in ("uint32", "int32"), msg

                self.desc_pointer_param_indices |= indices

    def visit_let_stmt(self, let_stmt):
        value = let_stmt.value
        if not is_pointer(value):
            super().visit_let_stmt(let_stmt)
            return

        # Visit value to collect "_call2ret_ptr_associated_args".
        super().visit_expr(value)

        chain_items = {_ChainItem(self._cur_blk_id, x) for x in self._get_origin_vars_of_ptr(value)}
        self._alias2ptrs[let_stmt.var] = {self._cur_blk_id: chain_items}

        super().visit_stmt(let_stmt.body)
        self._alias2ptrs.pop(let_stmt.var)

    def visit_if_then_else(self, ite):
        self.visit_expr(ite.condition)

        before_if_blk_id = self._cur_blk_id

        then_blk_id = self._cur_blk_id + 1
        for _, v in self._alias2ptrs.items():
            v[then_blk_id] = v[before_if_blk_id].copy()
        self._cur_blk_id = then_blk_id

        self.visit_stmt(ite.then_case)

        else_blk_id = then_blk_id + 1
        for _, v in self._alias2ptrs.items():
            v[else_blk_id] = v[before_if_blk_id].copy()
        self._cur_blk_id = else_blk_id

        if ite.else_case:
            self.visit_stmt(ite.else_case)

        after_if_blk_id = else_blk_id + 1
        for _, v in self._alias2ptrs.items():
            v[after_if_blk_id] = v[then_blk_id] | v[else_blk_id]
        self._cur_blk_id = after_if_blk_id

    def visit_for(self, for_op):
        self.visit_expr(for_op.min)
        self.visit_expr(for_op.extent)

        before_blk_id = self._cur_blk_id

        inner_blk_id = self._cur_blk_id + 1
        for _, v in self._alias2ptrs.items():
            v[inner_blk_id] = v[before_blk_id].copy()
        self._cur_blk_id = inner_blk_id

        self.visit_stmt(for_op.body)

        after_blk_id = inner_blk_id + 1
        for _, v in self._alias2ptrs.items():
            v[after_blk_id] = v[before_blk_id] | v[inner_blk_id]
        self._cur_blk_id = after_blk_id

    def visit_while(self, while_op):
        self.visit_expr(while_op.condition)

        before_blk_id = self._cur_blk_id

        inner_blk_id = self._cur_blk_id + 1
        for _, v in self._alias2ptrs.items():
            v[inner_blk_id] = v[before_blk_id].copy()
        self._cur_blk_id = inner_blk_id

        self.visit_stmt(while_op.body)

        after_blk_id = inner_blk_id + 1
        for _, v in self._alias2ptrs.items():
            v[after_blk_id] = v[before_blk_id] | v[inner_blk_id]
        self._cur_blk_id = after_blk_id


def extract_prim_func_info(ir_mod):
    """Get the information of the primitive function by analyzing the given TIR IRModule."""
    entry_funcs = tuple(x for x in ir_mod.functions.values() if x.attrs["tir.is_entry_func"])
    assert len(entry_funcs) == 1, "IRModule must and only can contains 1 entry function."
    entry_func = entry_funcs[0]
    assert isinstance(entry_func, tir.PrimFunc), "The entry function must be PrimFunc."

    params = entry_func.params
    ret = PrimFuncInfo(entry_func.attrs["global_symbol"], len(params))
    input_idx, output_idx = 0, 0
    analyser = _Analyser(entry_func, ir_mod)
    written_pointer_param_indices = analyser.written_pointer_param_indices
    desc_pointer_param_indices = analyser.desc_pointer_param_indices

    for i, param in enumerate(params):
        if param.dtype == "handle":
            dtype = param.type_annotation.element_type.dtype
            if i in desc_pointer_param_indices:
                info = ParamInfo(param.name, dtype, "descriptor")
            elif i in written_pointer_param_indices:
                info = ParamInfo(param.name, dtype, "output_tensor", tensor_idx=output_idx)
                output_idx += 1
            elif param.type_annotation.storage_scope == "global.1":
                info = ParamInfo(param.name, dtype, "const_tensor")
            else:
                info = ParamInfo(param.name, dtype, "input_tensor", tensor_idx=input_idx)
                input_idx += 1
        else:
            # Treat all parameters that aren't pointer as attribute in Compass IR.
            info = ParamInfo(param.name, param.dtype, "attr")

        ret.param_infos[i] = info

    return ret
