# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Get the associated data type of all mask nodes."""
from collections import defaultdict
from tvm import tir, ir


def _is_mask(x):
    if isinstance(x, tir.IntImm):
        # The scalar boolean literal is definitely not a mask, by the way, an instance of class
        # "tir.IntImm" can't be a key of dictionary because of its "__eq__" method.
        return False
    # Because the pass "AlignVectorWidthBySplit" may split out a vector whose
    # length is 1, so here we can't assume that mask must be a boolean vector.
    return x.dtype.is_bool


class _MaskDependenceAnalyzer(tir.StmtExprVisitor):
    def __init__(self):
        super().__init__()
        # The key is mask node, value is all other mask nodes that dependent by the key.
        #     a = S.const_mask("4TF")
        #     c = S.vand(a, b, mask=S.const_mask("8T"))
        # For the above example, the dictionary is something like below.
        # {a: {S.const_mask("4TF")}, c: {S.vand}, S.vand: {a, b, S.const_mask("8T")}}
        self._mask2dependencies = defaultdict(set)

    def visit_call(self, call):
        super().visit_call(call)

        if call.op == ir.Op.get("tir.reassign"):
            var, value = call.args
            if _is_mask(value):
                self._mask2dependencies[var].add(value)
            return

        if call.op != ir.Op.get("tir.call_extern"):
            return

        if not _is_mask(call):
            return  # Only need record the dependencies of the mask node.

        mask_args = tuple(x for x in call.args[1:] if _is_mask(x))
        self._mask2dependencies[call].update(mask_args)

    def visit_let_stmt(self, let_stmt):
        super().visit_let_stmt(let_stmt)

        var, value = let_stmt.var, let_stmt.value
        if _is_mask(value):
            self._mask2dependencies[var].add(value)

    def analyze(self, func):
        self.visit(func.body)
        return self._mask2dependencies


_arg1_funcs = ("vstore", "vstore_scatter", "vclt", "vcgt", "vcle", "vcge", "vceq", "vcneq")
_arg1_funcs += ("vnsr",)
_arg2_funcs = ("vmull", "vmulh", "__vdpa", "__vqdpa", "__vrpadd", "__vdot", "__vqdot")
_arg12_funcs = ("vzip",)
_arg23_funcs = ("vinv",)
_arg123_funcs = ("__vsel",)
_arg234_funcs = ("vand", "vor", "vxor")


def _get_mask_dtype_arg_indices(func_name):
    # The indices of the arguments that can help to determine the lanes of the mask argument.
    # "None" means the lanes of the mask argument is same as the return value.
    if func_name in _arg1_funcs:
        return (1,)
    if func_name in _arg2_funcs:
        return (2,)
    if func_name in _arg12_funcs:
        return (1, 2)
    if func_name in _arg23_funcs:
        return (2, 3)
    if func_name in _arg123_funcs:
        return (1, 2, 3)
    if func_name in _arg234_funcs:
        return (2, 3, 4)
    return None


def _change_for_call_if_needed(call, associated_dtype_of_args):
    func_name = call.args[0].value
    if func_name == "vzip" and call.args[3].value == "all":
        return associated_dtype_of_args.with_lanes(associated_dtype_of_args.lanes * 2)
    return associated_dtype_of_args


def _change_for_args_if_needed(call, associated_dtype_of_call):
    if not isinstance(call, tir.Call) or call.op != ir.Op.get("tir.call_extern"):
        return associated_dtype_of_call  # Will happened when propagating recursively.

    func_name = call.args[0].value
    if func_name == "vzip" and call.args[3].value == "all":
        return associated_dtype_of_call.with_lanes(associated_dtype_of_call.lanes // 2)
    return associated_dtype_of_call


class _Analyzer(tir.StmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.mask2associated_dtype = {}
        self._mask2dependencies = None
        self.mask_used_in_subfunc = {}  # key: mask, value: (gvar of subfunc, arg index)

    def _only_add_first(self, mask, associated_dtype):
        if mask in self.mask2associated_dtype:
            return
        self.mask2associated_dtype[mask] = associated_dtype

    def _get_associated_dtype(self, call):
        indices = _get_mask_dtype_arg_indices(call.args[0].value)
        if indices is None:  # Indicate the lanes of the mask argument is same as the return value.
            return_dtype = call.dtype
            return None if return_dtype.is_bool else return_dtype

        for idx in indices:
            arg = call.args[idx]
            dtype = arg.dtype
            if not dtype.is_bool:
                return _change_for_call_if_needed(call, dtype)
            # This argument also is a mask variable, the associated data type of it may already be
            # recorded or propagated.
            if arg in self.mask2associated_dtype:
                return _change_for_call_if_needed(call, self.mask2associated_dtype[arg])

        return None

    def _try_update_by_mask_var_assign(self, var, value):
        # Record the associated data type of the mask variables when they are assigned, so the
        # information can be propagated to the arguments of the mask operation functions, e.g.,
        # "vand".
        if _is_mask(value) and value in self.mask2associated_dtype:
            self._only_add_first(var, self.mask2associated_dtype[value])

    def visit_let_stmt(self, let_stmt):
        self.visit_expr(let_stmt.value)
        self._try_update_by_mask_var_assign(let_stmt.var, let_stmt.value)
        self.visit_stmt(let_stmt.body)

    def _recursive_propagate(self, associated_dtype, masks):
        for mask in masks:
            self._only_add_first(mask, associated_dtype)
            if mask not in self._mask2dependencies:
                continue

            # Have dependencies, need propagate recursively.
            dependency_associated_dtype = _change_for_args_if_needed(mask, associated_dtype)
            self._recursive_propagate(dependency_associated_dtype, self._mask2dependencies[mask])

    def visit_call(self, call):
        super().visit_call(call)

        if call.op == ir.Op.get("tir.reassign"):
            self._try_update_by_mask_var_assign(*call.args)
            return

        if isinstance(call.op, ir.GlobalVar):
            for i, arg in enumerate(call.args):
                if _is_mask(arg):
                    self.mask_used_in_subfunc[arg] = (call.op, i)

        if call.op != ir.Op.get("tir.call_extern"):
            return

        mask_args = tuple(x for x in call.args[1:] if _is_mask(x))
        if len(mask_args) == 0:
            return

        associated_dtype = self._get_associated_dtype(call)
        if associated_dtype is None:
            # Indicate current function can't get the associated data type by
            # itself, e.g., boolean version of broadcast.
            return

        # Propagate the associated data type to all mask arguments.
        self._recursive_propagate(_change_for_args_if_needed(call, associated_dtype), mask_args)

        # If current node is a mask generate function, e.g., "vcgt", the associated data type also
        # should be recorded to itself, because it maybe inline.
        if call.dtype.is_bool:
            self._only_add_first(call, associated_dtype)

    def analyze(self, func):
        self._mask2dependencies = _MaskDependenceAnalyzer().analyze(func)
        self.visit(func.body)


def get_mask_associated_dtype(mod):
    """Get the associated data type of all mask nodes for all functions in module.

    The return value is a map where key is the global var in module and the value is a map for each
    corresponding function, where key is the mask node and the value is its associated data type,
    the associated data type maybe is its user's data type, e.g., "tir.const_pred" node in "vadd",
    maybe is the creator's data type of the argument that determine the lanes of the mask argument,
    e.g., "tir.const_pred" node in "vand".
    This mapping information not only can be used for splitting and padding mask nodes according to
    the hardware vector width, but also can be used for padding "False" on high position of the
    "tir.const_pred" nodes.

    Precondition
      1. The mask nodes bind to the same variable must have the same length, need to be guaranteed
         by parser.
      2. The different indirect usages of the same mask node should have the same data type, need to
         be checked by this function.
      3. Only should be called after pass "LowerStandard".
      4. The assignment of all mask nodes must be represented by let statement or "tir.reassign",
         i.e., defining a bool array through "S.alloc" isn't allowed, need to be guaranteed by
         script APIs.
    """
    ret = {}
    gvar2analyzer = {}

    # 1. Analyze each func to get all mask dtype map except masks used in subfuntions.
    for gvar, func in mod.functions.items():
        analyzer = _Analyzer()
        analyzer.analyze(func)
        gvar2analyzer[gvar] = analyzer

    def _recursive_update_dtype(mask, cur_analyzer, subfunc_gvar, index):
        """Update mask dtype recursively which used in subfunctions."""
        sub_analyzer = gvar2analyzer[subfunc_gvar]
        param = mod[subfunc_gvar].params[index]
        if param in sub_analyzer.mask2associated_dtype:
            asso_dtype = sub_analyzer.mask2associated_dtype[param]
        else:
            assert param in sub_analyzer.mask_used_in_subfunc
            subfunc_gvar, param_id = sub_analyzer.mask_used_in_subfunc[param]
            asso_dtype = _recursive_update_dtype(param, sub_analyzer, subfunc_gvar, param_id)
        cur_analyzer._recursive_propagate(asso_dtype, [mask])
        return asso_dtype

    # 2. Get mask dtype map used in subfuntions.
    for gvar, analyzer in gvar2analyzer.items():
        for mask, (subfunc_gvar, param_id) in analyzer.mask_used_in_subfunc.items():
            if mask not in analyzer.mask2associated_dtype:
                _recursive_update_dtype(mask, analyzer, subfunc_gvar, param_id)
        ret[gvar] = analyzer.mask2associated_dtype
    return ret
