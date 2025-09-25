# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=eval-used
"""The parser part of Zhouyi Compass script APIs."""
import ast
import inspect
import textwrap
import functools
from tvm import ir, tir, script, IRModule
from tvm.script import tir as T
from .pysim import PyPrimFunc, PySimInfo


def prim_func(func=None, is_entry=False, is_inline=None):  # pylint: disable=unused-argument
    """Decorator for prim_func definitions.

    Parameters
    ----------
    func : Callable
        The function to be parsed as prim func.

    is_entry: Optional[bool]
        Whether the function is entry kernel. If this option is not set, it will automatically
        perform function dependency analysis to infer which function is the entry kernel function.

    is_inline : Optional[bool]
        Whether the function is treated as an inline function in codegen.

        - None: codegen will not add any attribute for the current function.
        - True: codegen will add "_CLC_INLINE" attr for the current function.
        - False: codegen will add "_CLC_NOINLINE" attr for the current function.

    Examples
    --------
    .. code-block:: python

        @S.prim_func
        def func(xxx):
            xxx

        @S.prim_func(is_inline=False)
        def utils_func(xxx):
            xxx

        @S.prim_func(is_entry=True)
        def kernel_func(xxx):
            utils_func(xxx)
            func(xxx)
    """
    assert is_inline in (None, True, False), 'Invalid value of parameter "is_inline".'

    attrs = {}
    if is_inline is not None:
        attrs["is_inline"] = is_inline

    def _decorator(myf):
        return functools.wraps(myf)(PyPrimFunc(myf, attrs))

    setattr(_decorator, "dispatch_token", "tir")
    return _decorator(func) if func else _decorator


setattr(prim_func, "dispatch_token", "tir")


def macro(*args, hygienic=True):
    """Decorator for macro definitions.

    Parameters
    ----------
    hygienic: Optional[bool]
        Specifies whether the macro is hygienic or not.

        A macro is hygienic if all symbols used in the macro's body are resolved
        to values from the location of the macro definition. A non-hygienic macro
        will have its symbols resolved to values at the time of macro use.

    Examples
    --------
    .. code-block:: python

        from tvm.compass.dsl import script as S

        x_value = 128

        @S.macro(hygienic=True)
        def static_capture(A, B):
            B[x_value] = A[x_value]     ## x_value binds to 128

        @S.macro(hygienic=False)
        def dynamic_capture(A, B):
            B[x_value] = A[x_value]     ## x_value will bind at the time of use


        @S.prim_func
        def use1(A: S.ptr("fp32", "global"), B: S.ptr("fp32", "global")):
            for x_value in range(10):
                static_capture(A, B)    ## Produces B[128] = A[128]

        @S.prim_func
        def use2(A: S.ptr("fp32", "global"), B: S.ptr("fp32", "global")):
            for x_value in range(10):
                dynamic_capture(A, B)   ## Produces B[x_value] = A[x_value]

    See Also
    --------
    - :doc:`../how_to_guides/how_to_use_macros`
    """

    def _decorator(*myf_args):
        tir_macro = T.macro(*myf_args, hygienic=hygienic)

        @functools.wraps(myf_args[0])
        def _wrapper(*func_args, **kwargs):
            assert PySimInfo.current is None, "PySim does not support macro."
            return tir_macro(*func_args, **kwargs)

        return _wrapper

    return _decorator if len(args) == 0 else _decorator(*args)


def parse_to_prim_func(func):
    """Parse to TensorIR PrimFunc through TVM Script parser."""
    msg = f'The function "{func.__module__}.{func.__name__}" must be decorated by "S.prim_func".'
    assert isinstance(func, PyPrimFunc), msg

    ret = T.prim_func(func.py_func, check_well_formed=False)
    return ret.with_attr(func.attrs) if len(func.attrs) != 0 else ret


class FindPrimFuncCallee(ast.NodeVisitor):
    """To find other prim_func is called in input prim_func."""

    def __init__(self, extra_vars):
        self.callee = []
        self._exclude = [range]
        self.extra_vars = extra_vars

    def visit_Call(self, node):  # pylint: disable=invalid-name
        """
        To find funtion callee, we visit ast CallNode.
        If the callee is prim_func, then the callee is subroutine.
        Otherwise it is a python function to eval.
        """
        exe = compile(ast.Expression(node.func), filename="<ast>", mode="eval")
        try:
            func = eval(exe, self.extra_vars)
            if (
                func not in self._exclude
                and func not in self.callee
                and isinstance(func, PyPrimFunc)
            ):
                self.callee.append(func)
        except (NameError, ValueError, RuntimeError):
            pass
        self.generic_visit(node)


def find_dependency(py_func):
    """
    This function finds all prim_func that have been
    called by input function. Do depth-first traversal.
    """

    def _find_prim_func_callee(func):
        """
        This function finds all prim_func that have been
        called by input function. Not do depth traversal.
        """
        extra_vars = script.parser._core.utils.inspect_function_capture(func.py_func)
        visitor = FindPrimFuncCallee(extra_vars)

        def _remove_indent(source):
            """remove indent"""
            source = textwrap.dedent(source)
            lines = source.split("\n")
            return "\n".join(lines)

        source = _remove_indent(inspect.getsource(func))
        func_ast = ast.parse(source)
        visitor.visit(func_ast)
        return visitor.callee

    # DFS to collect all used prim_func
    collect_functions = []
    stack = _find_prim_func_callee(py_func)
    while len(stack) > 0:
        if stack[-1] not in collect_functions:
            children_callee = _find_prim_func_callee(stack[-1])
            if py_func in children_callee:
                msg = f"{stack[-1].func.__name__} calls {py_func.py_func.__name__}, current not "
                msg += "support entry function as subroutines."
                raise RuntimeError(msg)
            new_func = [
                func
                for func in children_callee
                if func not in stack and func not in collect_functions
            ]
            if len(new_func) == 0:
                # the last func does not call other new functions, add to collection
                collect_functions.append(stack.pop())
            else:
                stack = stack + new_func
        else:
            # already visited
            stack.pop()
    if py_func not in collect_functions:
        collect_functions.append(py_func)
    return collect_functions


class RefineGlobalVarMutator(tir.StmtExprMutator):
    """
    If subroutine-call happens, prim_func call a global_var.
    However the global_var is not the object contained in module.
    They only share the same name_hint.
    This mutator will correct it.
    """

    def __init__(self, global_vars):
        super().__init__()
        self._global_vars = {}
        for gvar in global_vars:
            name = str(gvar.name_hint)
            self._global_vars[name] = gvar

    def visit_call(self, call):
        """visit call to correct op"""
        update_call = super().visit_call(call)
        if isinstance(call.op, ir.GlobalVar):
            name = str(call.op.name_hint)
            if name in self._global_vars:
                op = self._global_vars[name]
                update_call = tir.Call(update_call.dtype, op, update_call.args, update_call.span)

        return update_call


def parse_to_module(py_func, name=None):
    """Parse the given Python function to IRModule which contains multi PrimFunc."""

    funcs = find_dependency(py_func)
    mod_dict = {}
    for func in funcs:
        prim_func_ = parse_to_prim_func(func)
        if func is py_func:
            prim_func_ = prim_func_.with_attr("tir.is_entry_func", True)
        else:
            prim_func_ = prim_func_.with_attr("tir.is_entry_func", False)
        func_name = name if name is not None and func is py_func else func.__name__
        prim_func_ = prim_func_.with_attr("global_symbol", func_name)
        mod_dict[func_name] = prim_func_
    mod = IRModule(mod_dict)
    global_vars = list(mod.functions.keys())

    mutator = RefineGlobalVarMutator(global_vars)
    for gvar, prim_func_ in mod.functions.items():
        prim_func_ = tir.PrimFunc(
            prim_func_.params,
            mutator.visit(prim_func_.body),
            prim_func_.ret_type,
            prim_func_.buffer_map,
            prim_func_.attrs,
            prim_func_.span,
        )
        mod.update_func(gvar, prim_func_)
    return mod
