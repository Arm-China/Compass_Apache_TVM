# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Find post process function."""
from collections import deque, defaultdict
from tvm import relax, ir
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.dpl import rewrite_call
from .pattern_rewrites import EliminateUselessPermuteDims
from ..utils import X86_DESIRED_LAYOUTS


@mutator
class ParamUpdator(PyExprMutator):
    """Create new params for post function."""

    def __init__(self, params, mod=None):
        super().__init__(mod)
        self.params = params
        self.var2new_var = {}

    def visit_var_def_(self, var):
        if var in self.params:
            new_var = relax.Var(var.name_hint, var.struct_info)
            self.var2new_var[var] = new_var
            return new_var
        return var

    def visit_dataflow_var_def_(self, var):
        if var in self.params:
            new_var = relax.Var(var.name_hint, var.struct_info)
            self.var2new_var[var] = new_var
            return new_var
        return var

    def visit_var_(self, var):
        return self.var2new_var.get(var, var)

    def visit_dataflow_var_(self, var):
        return self.var2new_var.get(var, var)


def get_all_vars_from_unused_chain(duchain):
    """Get vars from unused chain

    Param:
        duchain: Dict{Var: List[Var]}
            define-use chain.

    Return:
        unused_vars: List[Var]

    Example:
        duchain: {A:[B], B:[C], C:[], a:[b]}
        return: [C, B, A]
    """
    outputs = []
    # deep copy from src chain
    current_chain = {k: v[:] for k, v in duchain.items()}
    dependents_map = defaultdict(set)
    for parent, children in current_chain.items():
        for child in children:
            if child in current_chain:
                dependents_map[child].add(parent)

    queue = deque(k for k, v in current_chain.items() if len(v) == 0)

    while queue:
        var_def = queue.popleft()
        if var_def not in current_chain:
            continue

        # collect the var and delete in chain
        outputs.append(var_def)
        del current_chain[var_def]

        for depend in dependents_map.get(var_def, set()):
            new_uses = [x for x in current_chain[depend] if x != var_def]
            current_chain[depend] = new_uses
            if len(new_uses) == 0 and depend not in queue:
                queue.append(depend)

    return outputs


def divide_post_process_function(ir_mod, check):
    """
    Cut the main function into 2 and the second
    part would be merged as global function
    """
    ir_mod = relax.transform.TopologicalSort()(ir_mod)
    blocks = ir_mod["main"].body.blocks
    if len(blocks) > 1 or not isinstance(blocks[0], relax.DataflowBlock):
        return ir_mod

    var2val = relax.analysis.get_var2val(ir_mod["main"])
    bindings = list(blocks[0].bindings)
    divide_bindings = [x for x in bindings[::-1] if check(var2val, x.value)]
    divide_vars = [x.var for x in divide_bindings]
    num = max(bindings.index(x) for x in divide_bindings) + 1
    duchain = relax.analysis.udchain(relax.DataflowBlock(bindings[:num]))
    duchain = {k: v for k, v in duchain.items() if k not in divide_vars}
    unused_vars = get_all_vars_from_unused_chain(duchain)
    remain_bindings = [x for x in bindings[:num] if x.var not in unused_vars]

    bb = relax.BlockBuilder()  # pylint: disable=invalid-name
    post_bindings = [x for x in bindings if x not in remain_bindings]
    post_seqe = relax.SeqExpr([relax.DataflowBlock(post_bindings)], ir_mod["main"].body.body)
    post_args = list(relax.analysis.free_vars(post_seqe))
    post_func = relax.Function(post_args, bb.normalize(post_seqe))
    post_func = ParamUpdator(post_args).visit_expr(post_func)
    post_mod = ir.IRModule.from_expr(post_func)
    # Do some optimize for post function
    post_mod = relax.transform.ConvertLayout(X86_DESIRED_LAYOUTS)(post_mod)
    post_mod = relax.transform.FoldConstant()(post_mod)
    post_func = rewrite_call(*EliminateUselessPermuteDims().pr, post_mod["main"])

    name = "post_process_func"
    # Avoid doing fuse ops in post function.
    post_func = post_func.with_attrs(
        {"global_symbol": name, "kComposite": "post_func", "Primitive": 1}
    )

    post_func_var = bb.add_func(post_func, name)
    out_var = relax.Var("out")
    bind = relax.VarBinding(out_var, post_func_var(*post_args))
    remain_bindings.append(bind)
    remain_block = relax.DataflowBlock(remain_bindings)
    remain_seqe = relax.SeqExpr([remain_block], out_var)
    remain_func = relax.Function(ir_mod["main"].params, bb.normalize(remain_seqe))
    remain_func = remain_func.with_attrs(ir_mod["main"].attrs)
    bb.add_func(remain_func, "main")
    return bb.get()


@ir.transform.module_pass(opt_level=0)
class GetPostProcessFunction:
    """Module pass to find post process."""

    def __init__(self, check_func):
        self.check = check_func

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        if self.check is None:
            return ir_mod
        if len(ir_mod.get_global_vars()) != 1 or ir_mod.get_global_vars()[0].name_hint != "main":
            return ir_mod
        divided_mod = divide_post_process_function(ir_mod, self.check)
        return divided_mod
