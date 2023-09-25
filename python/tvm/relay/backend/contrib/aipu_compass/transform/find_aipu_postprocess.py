# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""Try to find postprocess of graph"""
import uuid
import tvm
from tvm import relay, ir


def get_unique():
    return str(uuid.uuid4().fields[-1])[:5]


def _aipu_func_default_check(node):
    """
    This is a default check function to partition aipu graph
    """
    if not isinstance(node, relay.Call):
        return False

    if relay.ty.is_dynamic(node.checked_type):
        return False

    if isinstance(node.op, ir.op.Op):
        return False
    if "Composite" in node.op.attrs and "aipu_compass" in node.op.attrs["Composite"]:
        return True
    return False


def find_main_dependency_nodes(expr):
    """return a dict which all expr depended by input expr are as keys
    and the value is the set of expr which key depends on

    Parameters
    ----------
    expr : tvm.relay.Expr
        Expression

    Returns
    -------
    ret : dict and all expr depended by input expr are as keys
    and the value is the set of expr which key depends on
    """

    class DataflowVisitor(relay.ExprVisitor):
        """
        Helper Visitor to find dataflow nodes(call, tuple, tuple_getitem)
        """

        def __init__(self):
            super(DataflowVisitor, self).__init__()
            self.nodes = dict()

        def get_nodes(self, expr):
            if isinstance(expr, tvm.IRModule):
                self.visit(expr["main"].body)
            elif isinstance(expr, relay.Function):
                self.visit(expr.body)
            else:
                self.visit(expr)
            return self.nodes

        def visit_tuple(self, tup):
            for arg in tup.fields:
                self.visit(arg)
            dep = set([tup])
            for arg in tup.fields:
                if arg in self.nodes:
                    dep = dep | self.nodes[arg]
            self.nodes[tup] = dep

        def visit_call(self, call):
            # if it's a function call instead of op call
            # not visit the body
            for arg in call.args:
                self.visit(arg)
            dep = set([call])

            for arg in call.args:
                if arg in self.nodes:
                    dep = dep | self.nodes[arg]
            self.nodes[call] = dep

        def visit_var(self, var):
            self.nodes[var] = set([var])

        def visit_tuple_getitem(self, t):
            self.visit(t.tuple_value)
            self.nodes[t] = set([t]) | self.nodes[t.tuple_value]

    return DataflowVisitor().get_nodes(expr)


def find_main_dependency_stack(expr):
    """return a list in which all expr are depended by input expr
    and the order is topological sorted

    Parameters
    ----------
    expr : tvm.relay.Expr
        Expression

    Returns
    -------
    ret : List of relay.Expr
    """

    class DataflowVisitor(relay.ExprVisitor):
        """
        Helper Visitor to find dataflow nodes(call, tuple, tuple_getitem)
        """

        def __init__(self):
            super(DataflowVisitor, self).__init__()
            self.nodes = []

        def get_nodes(self, expr):
            if isinstance(expr, tvm.IRModule):
                self.visit(expr["main"].body)
            elif isinstance(expr, relay.Function):
                self.visit(expr.body)
            else:
                self.visit(expr)
            return self.nodes

        def visit_tuple(self, tup):
            for arg in tup.fields:
                self.visit(arg)
            self.nodes.append(tup)

        def visit_call(self, call):
            # if it's a function call instead of op call
            # not visit the body
            for arg in call.args:
                self.visit(arg)
            self.nodes.append(call)

        def visit_var(self, var):
            self.nodes.append(var)

        def visit_tuple_getitem(self, t):
            self.visit(t.tuple_value)
            self.nodes.append(t)

    return DataflowVisitor().get_nodes(expr)


class RenameVarConvertor(relay.ExprMutator):
    """
    rename the params name
    """

    def __init__(self, var_dict):
        super(RenameVarConvertor, self).__init__()
        self.var_dict = var_dict
        self.new_vars = dict()

    def visit_var(self, var):
        if var in self.var_dict:
            new_var = relay.Var(f"postprocess_var_{get_unique()}", var.checked_type)
            self.new_vars[var] = new_var
            return new_var
        return var


@tvm.ir.transform.module_pass(opt_level=0)
class DividePostProcessFunction:
    """
    Module pass to cut the main function into 2 and the second
    part would be merged as global function
    """

    def __init__(self, check_func=_aipu_func_default_check):
        self.check = check_func

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """
        Function to transform module
        """

        if self.check is None:
            return mod

        update_mod = mod
        all_nodes = find_main_dependency_stack(update_mod["main"].body)
        dep_nodes = find_main_dependency_nodes(update_mod["main"].body)
        stack = all_nodes
        dependencies = set()

        while len(stack) > 0:
            size = len(stack)
            for i in range(size - 1, -1, -1):

                if self.check(stack[i]):
                    dependency = set(dep_nodes[stack[i]])
                    dependencies = dependencies | dependency
                    stack = [node for node in stack if node not in dependencies]
                    break
            if size == len(stack):
                break

        post_nodes = [node for node in all_nodes if node not in dependencies]
        main_vars = relay.analysis.free_vars(update_mod["main"].body)
        params = []
        args = []

        if len(post_nodes) == 0:
            return mod
        if update_mod["main"].body not in post_nodes:
            return mod

        class VarConvertor(relay.ExprMutator):
            """
            Helper class to change the call args into var
            """

            _check = lambda arg: arg in dependencies or arg in main_vars

            def visit_call(self, call):
                if call not in post_nodes:
                    return call

                new_call = super().visit_call(call)

                if all([not VarConvertor._check(arg) for arg in new_call.args]):
                    return new_call

                call_args = []
                for arg in new_call.args:
                    if VarConvertor._check(arg):
                        if arg not in args:
                            args.append(arg)
                            var = relay.Var(f"postprocess_var_{get_unique()}", arg.checked_type)
                            params.append(var)
                        idx = args.index(arg)
                        call_args.append(params[idx])
                    else:
                        call_args.append(arg)
                return relay.Call(
                    new_call.op, call_args, new_call.attrs, new_call.type_args, new_call.span
                )

            def visit_tuple(self, tup):
                if tup not in post_nodes:
                    return tup

                new_tuple = super().visit_tuple(tup)

                if all([not VarConvertor._check(arg) for arg in new_tuple.fields]):
                    return new_tuple

                fields_args = []
                for arg in new_tuple.fields:
                    if VarConvertor._check(arg):
                        if arg not in args:
                            args.append(arg)
                            var = relay.Var(f"postprocess_var_{get_unique()}", arg.checked_type)
                            params.append(var)
                        idx = args.index(arg)
                        fields_args.append(params[idx])
                    else:
                        fields_args.append(arg)
                return relay.Tuple(fields_args, new_tuple.span)

            def visit_var(self, var):
                if var not in post_nodes:
                    return var

                var = super().visit_var(var)

                if not VarConvertor._check(var):
                    return var

                if var not in args:
                    args.append(var)
                    new_var = relay.Var(f"postprocess_var_{get_unique()}", var.checked_type)
                    params.append(new_var)
                idx = args.index(var)
                return params[idx]

        fmain = VarConvertor().visit(update_mod["main"].body)
        free_vars = list(relay.analysis.free_vars(fmain))

        if len(free_vars) != len(params):
            # TODO(Yuchou Gan) unexpected situation
            if len(free_vars) < len(params):
                return mod
            fvars = [var_expr for var_expr in free_vars if var_expr not in params]
            if not all([var in main_vars for var in fvars]):
                return mod
            converter = RenameVarConvertor(fvars)
            fmain = converter.visit(fmain)
            new_vars = converter.new_vars
            for var in new_vars:
                params.append(new_vars[var])
                args.append(var)

        # To avoid annotate target for postprocess function
        # here hacked set the "Composite" attr as "default.XXX"
        new_func = relay.Function(params, fmain).with_attr("Composite", "default.default")
        mod_ = tvm.IRModule.from_expr(new_func)
        for gvar in update_mod.functions:
            if gvar.name_hint != "main":
                mod_.update_func(gvar, update_mod[gvar])
        new_func = relay.transform.InferType()(mod_)["main"].with_attr(
            "Composite", "default.default"
        )

        gvar_name = f"postprocess_func_{get_unique()}"
        gvar = relay.GlobalVar(gvar_name)
        update_mod[gvar] = new_func
        fmain = relay.Call(gvar, args)

        new_func = relay.Function(update_mod["main"].params, fmain)
        for gvar in update_mod.functions:
            if gvar.name_hint == "main":
                update_mod.update_func(gvar, new_func)
        return relay.transform.InferType()(update_mod)


@tvm.ir.transform.module_pass(opt_level=0)
class PartitionFunctionsToTwo:
    """
    Module to partition mod main function to two function by max_nodes
    """

    def __init__(self, max_nodes):
        self.max_nodes = max_nodes

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """
        Function to transform module
        """

        all_nodes = find_main_dependency_stack(mod)
        if len(all_nodes) <= self.max_nodes:
            return mod
        depends_count = dict()
        max_depends = len(all_nodes)
        for idx, expr in enumerate(all_nodes):
            depends_count[expr] = idx + 1

        max_nodes = self.max_nodes

        def f_check(expr):
            if expr in depends_count and max_depends - max_nodes > depends_count[expr]:
                return True
            return False

        return DividePostProcessFunction(f_check)(mod)  # noqa pylint: disable=not-callable


@tvm.ir.transform.module_pass(opt_level=0)
class LimitPostFunctionScale:
    """
    Module to limit postprocess function scale
    """

    def __init__(self, max_nodes):
        self.max_nodes = max_nodes

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """
        Function to transform module
        """

        postprocess_var = mod["main"].body.op
        if not (
            isinstance(postprocess_var, relay.GlobalVar)
            and "postprocess_func_" in postprocess_var.name_hint
        ):
            return mod

        postprocess_fun = mod[postprocess_var]
        used_gvar = []

        def fvisit(expr):
            if isinstance(expr, relay.GlobalVar):
                used_gvar.append(expr)

        relay.analysis.post_order_visit(postprocess_fun.body, fvisit)
        used_gvar_name = [gvar.name_hint for gvar in used_gvar]

        update_mod = tvm.IRModule.from_expr(postprocess_fun)
        for gvar in used_gvar:
            update_mod[gvar] = mod[gvar]

        main_func_params = update_mod["main"].params

        fmain = RenameVarConvertor(main_func_params).visit(update_mod["main"])
        update_mod["main"] = fmain
        update_mod = relay.transform.InferType()(update_mod)
        cur_global_funcs = len(update_mod.functions)
        pre_global_funcs = 0

        max_cut = 10
        for _ in range(max_cut):
            pass0 = PartitionFunctionsToTwo(self.max_nodes)
            update_mod = pass0(update_mod)  # noqa pylint: disable=not-callable
            pre_global_funcs = cur_global_funcs
            cur_global_funcs = len(update_mod.functions)
            if cur_global_funcs == pre_global_funcs:
                break

        update_mod = relay.transform.InferType()(update_mod)
        for gvar in update_mod.functions:
            if gvar.name_hint not in used_gvar_name and gvar.name_hint != "main":
                mod.update_func(gvar, update_mod[gvar])

        func = relay.Function(update_mod["main"].params, update_mod["main"].body).with_attr(
            "Composite", "default.default"
        )
        mod.update_func(postprocess_var, func)
        mod = relay.transform.InferType()(mod)
        return mod


@tvm.ir.transform.module_pass(opt_level=0)
class GetPostProcessFunction:
    """
    Module pass to find postprocess
    """

    def __init__(self, check_func=_aipu_func_default_check):
        self.check = check_func
        self.max_nodes = 4000

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """
        Function to transform module
        """
        if self.check is None:
            return mod

        mod = relay.transform.InferType()(mod)
        passes = [DividePostProcessFunction(self.check), LimitPostFunctionScale(self.max_nodes)]
        for update_pass in passes:
            mod = update_pass(mod)
        return mod
