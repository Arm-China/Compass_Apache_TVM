# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
"""Evaluate non primitive call that all args are constant."""
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    is_if,
    is_var,
    is_tuple,
    wildcard,
)
from tvm.relay import ExprMutator
from .find_aipu_postprocess import get_unique
from .unroll_let_loop_in_main import ExprReplace, find_expr


def get_let_stack(exprssion):
    return find_expr(exprssion, lambda expr: isinstance(expr, relay.Let))


def well_formed_var(let):
    convert_dict = dict()
    new_let_var = relay.Var(let.var, let.var.type_annotation)
    for var in let.value.params:
        convert_dict[var] = relay.Var(var.name_hint + "_" + get_unique() + "_var", var.checked_type)
    convert_dict[let.var] = new_let_var
    new_let = ExprReplace(convert_dict).visit(let)
    return new_let


class ChangeVarNameMutator(ExprMutator):
    def visit_var(self, var):
        return relay.Var(var.name_hint + "_" + get_unique(), var.type_annotation)


def relay_verify_let_params(mod, let):
    """
    Verify the let loop parameters,
    makes sure let body not use parameters that out of body scope
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

    depends_params = list(relay.analysis.free_vars(relay.Tuple(func_body.true_branch.args)))
    need_verify_vars = [var for var in depends_params if var not in let_value.params]

    if len(need_verify_vars) == 0:
        # no need to verify
        return mod

    let_var_calls = find_expr(
        let_body, lambda expr: isinstance(expr, relay.Call) and expr.op == let.var
    )
    if len(let_var_calls) == 0:
        let_var_calls = find_expr(
            mod["main"], lambda expr: isinstance(expr, relay.Call) and expr.op == let
        )

    if len(let_var_calls) != 1:
        return mod

    convert_var_dict = dict()
    var_checked_type = []

    for var in let_value.params:
        convert_var_dict[var] = relay.Var(var.name_hint + get_unique() + "_var", var.checked_type)

    for var in need_verify_vars:
        convert_var_dict[var] = relay.Var(get_unique() + "_var", var.checked_type)
        var_checked_type.append(var.checked_type)

    fn_type = func_body.true_branch.op.checked_type
    fn_type = tvm.ir.type.FuncType(
        list(fn_type.arg_types) + var_checked_type,
        tvm.ir.type.TupleType(list(fn_type.ret_type.fields) + var_checked_type),
    )
    new_while_var = relay.Var(get_unique() + "_fn_var", fn_type)

    true_call_args = list(func_body.true_branch.args) + need_verify_vars
    ret_true_branch = relay.Call(new_while_var, true_call_args)
    ret_true_branch = ExprReplace(convert_var_dict).visit(ret_true_branch)
    ret_false_branch = relay.Tuple(list(convert_var_dict.values()))

    new_cond = ExprReplace(convert_var_dict).visit(func_body.cond)

    func_if = relay.If(new_cond, ret_true_branch, ret_false_branch)
    func_vars = list(convert_var_dict.values())
    new_fn = relay.Function(func_vars, func_if)

    new_let = relay.Let(new_while_var, new_fn, new_while_var)
    let_call_dict = dict()
    call = let_var_calls[0]
    new_call = None
    if call.op == let:
        new_call = relay.Call(new_let, list(call.args) + need_verify_vars)
    elif call.op == let.var:
        new_call = relay.Call(new_while_var, list(call.args) + need_verify_vars)

    get_items = find_expr(
        fmain, lambda expr: isinstance(expr, relay.TupleGetItem) and expr.tuple_value == call
    )
    if new_call is not None:
        if not relay.analysis.well_formed(new_call) and relay.analysis.well_formed(new_let):
            new_let = ChangeVarNameMutator().visit(new_let)
            new_call = relay.Call(new_let, list(new_call.args))

            if call.op == let:
                new_call = relay.Call(new_let, list(new_call.args))
            elif call.op == let.var:
                new_call = relay.Call(new_while_var, list(call.args) + need_verify_vars)

        for get_item in get_items:
            let_call_dict[get_item] = relay.TupleGetItem(new_call, get_item.index)
        let_call_dict[call] = new_call
        context_mod["main"] = ExprReplace(let_call_dict).visit(fmain)

    return context_mod


@tvm.ir.transform.module_pass(opt_level=0)
class VerifyLetLoopParams:
    """
    Sometimes let loop body use parameters out of let body scope,
    Put these parameters into Let loop function parameters
    """

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """try transform module"""

        update_mod = mod

        let_stack = get_let_stack(update_mod["main"].body)
        unchanged_let = []
        while len(let_stack) != 0:
            let = let_stack[0]
            new_mod = relay_verify_let_params(update_mod, let)
            if update_mod is new_mod:
                unchanged_let.append(let)
            update_mod = new_mod
            let_stack = get_let_stack(update_mod["main"].body)
            let_stack = [let for let in let_stack if let not in unchanged_let]
        return update_mod
