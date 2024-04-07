# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Evaluate non primitive call that all args are constant."""
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    is_if,
    is_var,
    is_tuple,
    wildcard,
)
from .evaluate_zero_free_args_call import vmobj_to_list


class ExprReplace(relay.ExprMutator):
    """Helper mutator to mutate recorded expression"""

    def __init__(self, convert_dict):
        super(ExprReplace, self).__init__()
        self.convert_dict = convert_dict

    def visti_expr(self, expr):
        if expr in self.convert_dict:
            return self.convert_dict[expr]
        return expr

    def visit_var(self, var):
        if var in self.convert_dict:
            return self.convert_dict[var]
        return var

    def visit_call(self, call):
        new_call = super().visit_call(call)
        if call in self.convert_dict:
            return self.convert_dict[call]
        return new_call

    def visit_let(self, let):
        new_let = super().visit_let(let)
        if let in self.convert_dict:
            return self.convert_dict[let]
        return new_let


def eval_expr(context_mod, expr, *inputs):
    """Helper function to evaluate expression"""

    update_mod = tvm.IRModule.from_expr(expr)
    fmain = update_mod["main"]
    update_mod = tvm.IRModule(context_mod.functions, context_mod.type_definitions)
    update_mod["main"] = fmain

    eval_args_func = relay.create_executor(
        kind="vm", mod=update_mod, device=tvm.cpu(0), target="llvm"
    ).evaluate()
    eval_val = vmobj_to_list(eval_args_func(*inputs))
    return eval_val


def find_expr(base_expr, filter_func):
    """helper to find expression in base_expr expr depends on the filter"""
    ret = []

    def fvisit(expr):
        if filter_func(expr):
            ret.append(expr)

    relay.analysis.post_order_visit(base_expr, fvisit)
    return ret


def relay_unroll_loop(mod, let, loop_max):
    """
    Unroll loop in let expr
    If loop_num is more than loop_max, the loop would not unroll
    """
    context_mod = tvm.IRModule(mod.functions, mod.type_definitions)
    fmain = context_mod["main"]
    let_value = let.value
    let_body = let.body
    if not isinstance(let_value, relay.Function):
        return mod

    func_body = let_value.body
    var = is_var(let.var.name_hint)
    vars_num = len(let_value.params)
    tuple_params = []
    for param in let_value.params:
        tuple_params.append(is_var(param.name_hint))
    var_params = [wildcard() for _ in range(vars_num)]
    matches = is_if(wildcard(), var(*var_params), is_tuple(tuple_params))
    match = matches.match(func_body)
    if not match:
        return mod

    if_cond_depends_params = list(relay.analysis.free_vars(func_body.cond))
    let_value_params = list(let_value.params)
    if_cond_depends_params_idx = [let_value_params.index(var) for var in if_cond_depends_params]

    prev_num = -1
    origin_cond_args = [func_body.true_branch.args[idx] for idx in if_cond_depends_params_idx]
    update_cond_depends_args = origin_cond_args
    cur_num = len(update_cond_depends_args)
    while prev_num != cur_num:
        prev_num = cur_num
        update_cond_depends_params = list(
            relay.analysis.free_vars(relay.Tuple(update_cond_depends_args))
        )
        update_cond_depends_params_idx = [
            let_value_params.index(var) for var in update_cond_depends_params
        ]
        update_cond_depends_args = [
            func_body.true_branch.args[idx] for idx in update_cond_depends_params_idx
        ]
        cur_num = len(update_cond_depends_args)

    if not all([param in update_cond_depends_params for param in if_cond_depends_params]):
        extra_params = [
            param for param in if_cond_depends_params if param not in update_cond_depends_params
        ]
        update_cond_depends_params = update_cond_depends_params + extra_params

        update_cond_depends_params_idx = [
            let_value_params.index(var) for var in update_cond_depends_params
        ]
        update_cond_depends_args = [
            func_body.true_branch.args[idx] for idx in update_cond_depends_params_idx
        ]

    let_var_calls = find_expr(
        let_body, lambda expr: isinstance(expr, relay.Call) and expr.op == let.var
    )
    if len(let_var_calls) == 0:
        let_var_calls = find_expr(
            mod["main"], lambda expr: isinstance(expr, relay.Call) and expr.op == let
        )

    if len(let_var_calls) != 1:
        return mod
    let_var_calls_rewrite = dict()
    call = let_var_calls[0]

    const_args = dict()
    for idx, arg in enumerate(call.args):
        free_vars = relay.analysis.free_vars(arg)
        if len(free_vars) == 0:
            const_args[idx] = arg
    # check loop depend params related arguments are all constant
    if not all([idx in const_args for idx in update_cond_depends_params_idx]):
        return mod

    ###############################
    # find how many times it loops
    ###############################
    _params = []
    _update_args = []
    _args = []
    for idx, param in enumerate(update_cond_depends_params):
        _params.append(param)
        _update_args.append(update_cond_depends_args[idx])
        func_index = update_cond_depends_params_idx[idx]
        _args.append(const_args[func_index])
    dtype = tvm.ir.tensor_type.TensorType([], "int32")
    loop_count = relay.Var("__unrollloop_counter__", dtype)

    func_var = relay.Var("__unrollloop_counter_function__")
    func_true_branch = relay.Call(func_var, _update_args + [loop_count + relay.const(1, "int32")])
    func_false_branch = relay.Tuple(_params + [loop_count])
    func_cond = func_body.cond
    func_if = relay.If(func_cond, func_true_branch, func_false_branch)
    func_obj = relay.Function(_params + [loop_count], func_if)
    new_let = relay.Let(func_var, func_obj, relay.Call(func_obj, _args + [relay.const(0, "int32")]))
    counter = relay.TupleGetItem(new_let, len(_params))
    try:
        loop_num = int(eval_expr(context_mod, counter).numpy())
    except RuntimeError:
        return mod

    if loop_num > loop_max:
        return mod

    #################
    # unroll the loop
    #################
    true_args = list(func_body.true_branch.args)
    true_args = relay.Tuple(true_args)

    var_rewrite_dict = dict()
    for var, arg in zip(let.value.params, true_args.fields):
        var_rewrite_dict[var] = arg

    ret_tuple = relay.Tuple(list(let.value.params))
    for loop_id in range(loop_num):
        if loop_id == 0:
            ret_tuple = true_args
            continue
        var_rewrite_dict = dict()
        for var, arg in zip(let.value.params, ret_tuple.fields):
            var_rewrite_dict[var] = arg
        ret_tuple = ExprReplace(var_rewrite_dict).visit(true_args)

    var_rewrite_dict = dict()
    for var, arg in zip(let.value.params, call.args):
        var_rewrite_dict[var] = arg
    unroll_tuple = ExprReplace(var_rewrite_dict).visit(ret_tuple)
    let_var_calls_rewrite[call] = unroll_tuple
    if call.op == let.var:
        new_let_body = ExprReplace(let_var_calls_rewrite).visit(let.body)
        var_rewrite_dict = dict()
        var_rewrite_dict[let] = new_let_body
        context_mod["main"] = ExprReplace(var_rewrite_dict).visit(fmain)
    else:
        context_mod["main"] = ExprReplace(let_var_calls_rewrite).visit(fmain)
    return context_mod


@tvm.ir.transform.module_pass(opt_level=0)
class UnrollLetLoopInMain:
    """
    unroll let loop in relay module main function
    """

    def __init__(self, loop_max=1):
        """If loop num is more than loop_max, the loop would not unroll"""

        self.loop_max = loop_max

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """try transform module"""

        update_mod = mod

        def get_let_stack(mod):
            return find_expr(mod["main"].body, lambda expr: isinstance(expr, relay.Let))

        let_stack = get_let_stack(update_mod)
        unchanged_let = []
        while len(let_stack) != 0:
            let = let_stack[0]
            new_mod = relay_unroll_loop(update_mod, let, self.loop_max)
            if update_mod is new_mod:
                unchanged_let.append(let)
            update_mod = new_mod
            let_stack = get_let_stack(update_mod)
            let_stack = [let for let in let_stack if let not in unchanged_let]
        return update_mod
