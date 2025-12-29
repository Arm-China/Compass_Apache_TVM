# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Check whether the specified function is bellwether of given IRModule."""
from tvm import ir, relax


@relax.expr_functor.visitor
class CallPathTracer(relax.PyExprVisitor):
    """Trace all of the call path of specified function in the given IRModule."""

    def __init__(self, ir_mod, target_func_name):
        super().__init__()
        self._ir_mod = ir_mod
        self._var2func = dict(ir_mod.functions.items())
        self._target_func_name = target_func_name
        self._cur_func = None
        self._call_stack = []
        self._call_paths = []

    def visit_var_binding_(self, binding):
        if isinstance(binding.value, relax.Function):
            # Indicate it is a local defined function, the call place of it must be somewhere below,
            # here just skip the visit of the function definition, and visit it after the
            # corresponding call is visited.
            self._var2func[binding.var] = binding.value
            return
        super().visit_var_binding_(binding)

    def _collect_call_path(self, callee):
        if not isinstance(callee, (ir.Op, ir.GlobalVar)):
            # The type here may be "relax.Var", it represent a local defined function.
            return
        callee_name = callee.name if isinstance(callee, ir.Op) else callee.name_hint
        if callee_name == self._target_func_name:
            self._call_paths.append(self._call_stack[:])

    def visit_call_(self, call):
        for arg in call.args:
            self.visit_expr(arg)

        self._call_stack.append((self._cur_func, call))
        self._collect_call_path(call.op)
        if call.op in self._var2func:
            self.visit_expr(self._var2func[call.op])
        self._call_stack.pop()

    def visit_function_(self, fn):
        old_value = self._cur_func
        self._cur_func = fn
        super().visit_function_(fn)
        self._cur_func = old_value

    def trace(self, entry_func_name="main"):
        """Get all of the call paths of specified function, each call path is
        composed of several invoke pair("func", "call"), where "func" is the
        caller's definition and "call" is the call node of callee."""
        self.visit_expr(self._ir_mod[entry_func_name])
        return self._call_paths


def _is_pure_forward(func, call):
    if len(func.params) != len(call.args):
        return False
    for param, arg in zip(func.params, call.args):
        if not param.same_as(arg):
            return False
    return True


def check_bellwether(ir_mod, codegen_name):
    """Check whether the function whose "Codegen" attribute equals to the given
    value is bellwether of given IRModule."""
    candidate_funcs = []
    for _, func in ir_mod.functions.items():
        attrs = func.attrs
        if attrs and "Codegen" in attrs and attrs.Codegen == codegen_name:
            candidate_funcs.append(func)

    if len(candidate_funcs) != 1:
        # It must not be the bellwether, if there is 0 or multiple function
        # definition, because we assume there isn't any unused function.
        return False

    candidate_func_name = candidate_funcs[0].attrs.global_symbol
    call_paths = CallPathTracer(ir_mod, candidate_func_name).trace()
    if len(call_paths) != 1:
        # It must not be the bellwether, if there is 0 or multiple function
        # call, because there is only one bellwether in a flock.
        return False

    call_path = call_paths[0]
    # For each invoke pair("func", "call") of the calling path, the candidate
    # function is the bellwether only if all of the "call" arguments from the
    # parameters of "func".
    for func, call in call_path:
        if not _is_pure_forward(func, call):
            return False

    return True
