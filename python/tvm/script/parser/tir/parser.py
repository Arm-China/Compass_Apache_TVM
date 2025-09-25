# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""The base parser for tir"""
#
# This file has been modified by Arm China team.
#
import contextlib
from functools import partial
from typing import Any
import numpy as np

import tvm
from tvm.ir import GlobalVar, PrimType, PointerType
from tvm.tir import Buffer, IterVar, PrimExpr, Var
from tvm.tir import Let, StringImm, Pointer, convert_to_prim_expr
from tvm.tir import reassign, vector_set_element
from tvm.tir.buffer import is_integer_indices

from ....compass import CompassInfo
from ...ir_builder import ir as I
from ...ir_builder import tir as T
from ...ir_builder.base import IRBuilder
from ...ir_builder.base import IRBuilderFrame as Frame
from .._core import Parser, dispatch, doc


def bind_with_value(self: Parser, node: doc.expr, var_name: str, value: Any) -> Any:
    """Value binding methods when parsing with statement.
    e.g. binding i, j, k with T.grid(128, 128, 128), when parsing
        with T.grid(128, 128, 18) as i, j, k.

    Parameters
    ----------
    self : Parser
        The current parser.

    node : doc.expr
        The doc AST expression node for error reporting.

    var_name : str
        The variable name.

    value : Any
        The value to be bound with.

    Returns
    -------
    res : Any
        The bound value.
    """
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            bind_with_value(self, node, f"{var_name}_{i}", v)
        return value
    elif isinstance(value, (Buffer, Var)):
        IRBuilder.name(var_name, value)
        return value
    else:
        self.report_error(node, f"Do not know how to bind type: {type(value)} in with statement")
        raise NotImplementedError


def bind_for_value(self: Parser, node: doc.expr, var_name: str, value: Any) -> Any:
    """Value binding methods when parsing for statement.
    e.g. binding i, j, k with T.grid(128, 128, 128), when parsing
        for i, j, k in T.grid(128, 128, 128).

    Parameters
    ----------
    self : Parser
        The current parser.

    node : doc.expr
        The doc AST expression node for error reporting.

    var_name : str
        The variable name.

    value : Any
        The value to be bound with.

    Returns
    -------
    res : Any
        The bound value.
    """
    if isinstance(value, (list, tuple, tvm.ir.Array)):
        for i, v in enumerate(value):
            bind_for_value(self, node, f"{var_name}_{i}", v)
        return value
    elif isinstance(value, Var):
        IRBuilder.name(var_name, value)
        return value
    else:
        self.report_error(node, f"Do not know how to bind type: {type(value)} in for statement")
        raise NotImplementedError


def _get_defined_vars_in_prim_func(var_table):
    ret = {}
    for var_name, values in var_table.name2value.items():
        if len(values) == 0:
            continue
        if var_name in var_table.frames[0].vars and len(values) == 1:
            continue  # Exclude those only defined outside of the PrimFunc.
        ret[var_name] = values[-1]
    return ret


def bind_assign_value(self: Parser, node: doc.expr, var_name: str, value: Any) -> Any:
    """Value binding methods when parsing assign statement.
    e.g. binding vi, vj, vk with T.axis.remap("SSR", [i, j, k]), when parsing
        vi, vj, vk = T.axis.remap("SSR", [i, j, k]).

    Parameters
    ----------
    self : Parser
        The current parser.

    node : doc.expr
        The doc AST expression node for error reporting.

    var_name : str
        The variable name.

    value : Any
        The value to be bound with.

    Returns
    -------
    res : Any
        The bound value.
    """
    if isinstance(value, T.meta_var):
        return value.value
    elif isinstance(value, (list, tuple)):
        if CompassInfo.current() is not None:
            return value

        for i, v in enumerate(value):
            bind_assign_value(self, node, f"{var_name}_{i}", v)
        return value
    elif isinstance(value, Frame):
        value.add_callback(partial(value.__exit__, None, None, None))
        res = value.__enter__()
        IRBuilder.name(var_name, res)
        return res
    elif isinstance(value, np.ndarray):
        u8_data = value.view("uint8")
        dtype = tvm.DataType(value.dtype)
        alloc_const_frame = T.allocate_const(u8_data, dtype="uint8", extents=u8_data.shape)
        alloc_const_frame.add_callback(partial(alloc_const_frame.__exit__, None, None, None))
        u8_buffer_var = alloc_const_frame.__enter__()
        u8_var_name = "u8_" + var_name
        IRBuilder.name(u8_var_name, u8_buffer_var)
        u8_ptr = Pointer(dtype, "constant", u8_buffer_var, 0)
        new_var = Var(var_name, PointerType(PrimType(dtype), "constant"))
        frame = T.LetStmt(u8_ptr, var=new_var)
        frame.add_callback(partial(frame.__exit__, None, None, None))
        frame.__enter__()
        return tvm.tir.decl_buffer(value.shape, dtype, var_name, new_var)
    elif isinstance(value, (Buffer, IterVar)) or (
        isinstance(value, Var) and not self.var_table.exist(value)
    ):
        is_compass = CompassInfo.current() is not None
        if is_compass and isinstance(value, Buffer) and value.name != "":
            self.report_error(node, "Unsupport reassign a buffer, maybe you can use pointer.")
        IRBuilder.name(var_name, value)
        return value
    elif isinstance(value, Let) and StringImm("define_size_var") == value.body:
        if var_name in _get_defined_vars_in_prim_func(self.var_table):
            self.report_error(node, f'Variable "{var_name}" is redefined.')
        # Use the size var in the let expression directly and rename it.
        var = value.var
        IRBuilder.name(var_name, var)

        frame = T.LetStmt(value.value, var=var)
        frame.add_callback(partial(frame.__exit__, None, None, None))
        frame.__enter__()
        return var
    elif isinstance(value, Pointer):
        # The object created on the right side of the equal sign has no name,
        # so bind a name to it, please refer to Interface "S.alloc".
        if isinstance(value.base, Var) and value.base.name == "":
            IRBuilder.name(f"{var_name}_buf", value.base)

        defined_vars = _get_defined_vars_in_prim_func(self.var_table)
        if var_name in defined_vars:
            ptr = defined_vars[var_name]
            if not isinstance(ptr, Pointer):
                msg = f'Type mismatch assignment: "{ptr.dtype}" vs. "{value.dtype}*".'
                self.report_error(node, msg)

            if ptr.dtype != value.dtype:
                msg = f'Type mismatch assignment: "{ptr.dtype}*" vs. "{value.dtype}*", need to do '
                msg += "the explicit type conversion for the right hand side(i.e., new value)."
                self.report_error(node, msg)

            T.evaluate(reassign(ptr.base, value))
            return ptr

        ptr = Pointer(value.dtype, value.scope, name=var_name)
        frame = T.LetStmt(value, var=ptr.base)
        frame.add_callback(partial(frame.__exit__, None, None, None))
        frame.__enter__()
        return ptr
    elif isinstance(value, str):
        msg = "Can't define string inside primitive function, please define it outside."
        self.report_error(node, msg)
        return None
    else:
        value = tvm.runtime.convert(value)
        value = convert_to_prim_expr(value)  # Convert the auxiliary node, e.g., BufferRegion.

        var = Var(var_name, value.dtype)
        defined_vars = _get_defined_vars_in_prim_func(self.var_table)
        # Not a explicit size var definition statement, if there isn't a existing variable have
        # the same name, then treat it as a definition, otherwise treat is as a assignment.
        if var_name in defined_vars:
            var = defined_vars[var_name]
            if isinstance(var, Pointer):
                msg = f'Type mismatch assignment: "{var.dtype}*" vs. "{value.dtype}".'
                self.report_error(node, msg)

            if value.dtype != var.dtype:
                msg = f'Type mismatch assignment: "{var.dtype}" vs. "{value.dtype}", need to do the'
                msg += " explicit type conversion for the right hand side(i.e., new value)."
                self.report_error(node, msg)

        frame = T.LetStmt(value, var=var)
        frame.add_callback(partial(frame.__exit__, None, None, None))
        frame.__enter__()
        return var


def find_decorator_annotation(node: doc.FunctionDef, annotation: str, default: bool = True) -> bool:
    """
    Check the value of given annotation (argument name) in the prim_func decorator.
    Returns the value of the annotation if present, otherwise giving the default value.
    """
    # look for the named argument in the prim_func decorator
    for dec in node.decorator_list:
        if not isinstance(dec, doc.Call) or dec.func.attr != "prim_func":
            continue
        for keyword in dec.keywords:
            if keyword.arg == annotation:
                return keyword.value.value
    return default


@dispatch.register(token="tir", type_name="For")
def visit_for(self: Parser, node: doc.For) -> None:
    """The for visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.For
        The doc AST for node.
    """
    for_frame = self.eval_expr(node.iter)
    if not isinstance(for_frame, T.frame.ForFrame):
        self.report_error(
            node.iter,
            "Expect the for loop to be one of the following: "
            "range, T.serial, T.grid, T.parallel, T.vectorized, T.unroll, T.thread_binding",
        )
    with self.var_table.with_frame():
        with for_frame as iters:
            is_compass = CompassInfo.current() is not None
            iter_vars = node.target.elts if isinstance(node.target, doc.Tuple) else (node.target,)
            for var in iter_vars:
                if isinstance(var, doc.Starred):
                    var = var.value
                if is_compass and var.id in self.var_table.get():
                    self.report_error(
                        node.target,
                        f"The iter var {var.id} of the for loop has been defined in outside scope, "
                        "please use another name.",
                    )
            self.eval_assign(target=node.target, source=iters, bind_value=bind_for_value)
            self.visit_body(node.body)


@dispatch.register(token="tir", type_name="While")
def visit_while(self: Parser, node: doc.While) -> None:
    """The while visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.While
        The doc AST while node.
    """
    with self.var_table.with_frame():
        cond = self.eval_expr(node.test)
        with T.While(cond):
            self.visit_body(node.body)


@dispatch.register(token="tir", type_name="Assign")
def visit_assign(self: Parser, node: doc.Assign) -> None:
    """The assign visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Assign
        The doc AST assign node.
    """
    if len(node.targets) != 1:
        self.report_error(node, "Consequential assignments like 'a = b = c' are not supported.")
    lhs = node.targets[0]

    if isinstance(lhs, doc.Tuple) and isinstance(node.value, doc.Tuple):

        class ValueUpdater(doc.NodeTransformer):
            """Update node value by temporary variable if the node will be used in later"""

            def __init__(self, parser, lhs_id, lhs_var) -> None:
                super().__init__()
                self.parser = parser
                self.lhs_id = lhs_id
                self.lhs_var = lhs_var

            def visit_Name(self, node):  # pylint: disable=invalid-name
                """Find the node need to be replaced and return new node with tmp_nodeID"""
                if node.id == self.lhs_id:
                    new_name = doc.Name(
                        "tmp_" + node.id,
                        node.ctx,
                        node.lineno,
                        node.col_offset,
                        node.end_lineno,
                        node.end_col_offset,
                    )
                    new_var = bind_assign_value(self.parser, new_name, new_name.id, self.lhs_var)
                    self.parser.var_table.add(new_name.id, new_var)
                    return new_name
                return node

        elt_num = len(lhs.elts)
        for i in range(elt_num - 1):
            assert isinstance(lhs.elts[i], doc.Name)
            lhs_id = lhs.elts[i].id
            if (
                lhs_id not in self.var_table.name2value.keys()
                or len(self.var_table.name2value[lhs_id]) == 0
            ):
                continue
            lhs_var = self.eval_expr(lhs.elts[i])
            for j in range(i + 1, elt_num):
                updater = ValueUpdater(self, lhs_id, lhs_var)
                node.value.elts[j] = updater.visit(node.value.elts[j])

    if isinstance(node.value, doc.Subscript):
        check_slices = []
        if isinstance(node.value.slice, doc.Slice):
            check_slices = [node.value.slice]
        elif isinstance(node.value.slice, doc.Tuple):
            for p in node.value.slice.elts:
                if isinstance(p, doc.Slice):
                    check_slices.append(p)
        for s in check_slices:
            if not s.step and s.upper:
                s.step = doc.Constant(
                    1,
                    None,
                    1,
                    1,
                    s.upper.lineno,
                    s.upper.end_col_offset + 1,
                    s.upper.lineno,
                    s.upper.end_col_offset + 2,
                )

    rhs = self.eval_expr(node.value)
    if isinstance(lhs, doc.Subscript):
        if isinstance(lhs.slice, doc.Tuple):
            indices = []
            for index in lhs.slice.elts:
                indices.append(self.eval_expr(index))
        else:
            indices = self.eval_expr(lhs.slice)

        if not is_integer_indices(indices):
            msg = "All indices should be integer type, please convert it explicitly."
            self.report_error(node, msg)

        target = self.eval_expr(lhs.value)
        if isinstance(target, Var):
            T.evaluate(vector_set_element(target, indices, rhs))
        else:
            T.buffer_store(target, rhs, indices)
    else:
        self.eval_assign(target=lhs, source=rhs, bind_value=bind_assign_value)


@dispatch.register(token="tir", type_name="AugAssign")
def visit_aug_assign(self: Parser, node: doc.AugAssign) -> None:
    """The augmented assign visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.AugAssign
        The doc AST augmented assign node.
    """
    lhs_pos = (
        node.target.lineno,
        node.target.col_offset,
        node.target.end_lineno,
        node.target.end_col_offset,
    )
    rhs_pos = (
        node.value.lineno,
        node.value.col_offset,
        node.value.end_lineno,
        node.value.end_col_offset,
    )
    node.target.ctx = doc.Load(*lhs_pos)
    with self.var_table.with_frame():
        lhs_name = "__tvm_tmp_value_aug_assign_lhs"
        rhs_name = "__tvm_tmp_value_aug_assign_rhs"
        lhs_expr = self.eval_expr(node.target)
        rhs_expr = self.eval_expr(node.value)
        self.var_table.add(lhs_name, lhs_expr)
        self.var_table.add(rhs_name, rhs_expr)
        op = doc.BinOp(
            doc.Name(lhs_name, doc.Load(*lhs_pos), *lhs_pos),
            node.op,
            doc.Name(rhs_name, doc.Load(*rhs_pos), *rhs_pos),
            *lhs_pos,
        )
        rhs = self.eval_expr(op)
    lhs = node.target
    lhs.ctx = doc.Store(*lhs_pos)
    if isinstance(lhs, doc.Subscript):
        if isinstance(lhs.slice, doc.Tuple):
            indices = []
            for index in lhs.slice.elts:
                indices.append(self.eval_expr(index))
        else:
            indices = [self.eval_expr(lhs.slice)]
        T.buffer_store(self.eval_expr(lhs.value), rhs, indices)
    else:
        self.eval_assign(target=lhs, source=rhs, bind_value=bind_assign_value)


@dispatch.register(token="tir", type_name="AnnAssign")
def visit_ann_assign(self: Parser, node: doc.AnnAssign) -> None:
    """The annotated assign visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.AnnAssign
        The doc AST annotated assign node.
    """
    lhs = node.target
    rhs = self.eval_expr(node.value)
    ann_var = self.visit_tvm_annotation(node.annotation)
    if not isinstance(ann_var, Var):
        self.report_error(node.annotation, "Annotation should be Var")
    self.eval_assign(target=lhs, source=ann_var, bind_value=bind_assign_value)
    frame = T.LetStmt(rhs, var=ann_var)
    frame.add_callback(partial(frame.__exit__, None, None, None))
    frame.__enter__()


@dispatch.register(token="tir", type_name="With")
def visit_with(self: Parser, node: doc.With) -> None:
    """The with visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.With
        The doc AST with node.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(self.var_table.with_frame())
        for item in node.items:
            frame = self.eval_expr(item.context_expr)
            if not isinstance(frame, Frame):
                self.report_error(
                    item.context_expr, "Invalid context expression in the with-statement."
                )
            rhs = stack.enter_context(frame)
            if item.optional_vars is not None:
                self.eval_assign(target=item.optional_vars, source=rhs, bind_value=bind_with_value)
        self.visit_body(node.body)


@dispatch.register(token="tir", type_name="FunctionDef")
def visit_function_def(self: Parser, node: doc.FunctionDef) -> None:
    """The function definition visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.FunctionDef
        The doc AST function definition node.
    """
    supplied_annotation = self.function_annotations
    func_annotation = supplied_annotation.get(node.name, {})
    privacy = find_decorator_annotation(node, "private", default=False)
    self.function_annotations = None
    with self.var_table.with_frame():
        self.var_table.add("range", T.serial)
        with T.prim_func(is_private=privacy):
            T.func_name(node.name)
            if node.returns is not None:
                try:
                    ret_type = self.eval_expr(node.returns)
                except Exception:  # pylint: disable=broad-except
                    ret_type = func_annotation.get("return", None)
                    if ret_type is None:
                        raise
                if hasattr(ret_type, "type_ann_func"):
                    ret_type = PrimType(ret_type.type_ann_func().dtype)
                elif isinstance(ret_type, Pointer):
                    ret_type = PointerType(PrimType(ret_type.dtype), ret_type.scope)
                elif callable(ret_type):
                    ret_type = PrimType(ret_type().dtype)
                T.func_ret(ret_type)
                self.ret_type = ret_type
            with self.with_dispatch_token("tir"):
                # TODO: handle different types of arguments:
                # - vararg: arg | None
                # - kwonlyargs: list[arg]
                # - kw_defaults: list[expr | None]
                # - kwarg: arg | None
                # - defaults: list[expr]
                # - posonlyargs: list[arg]
                for arg in node.args.args:
                    if arg.annotation is None:
                        self.report_error(arg, "Type annotation required for function parameters.")
                    try:
                        ann = self.eval_expr(arg.annotation)
                        if hasattr(ann, "type_ann_func"):
                            ann = ann.type_ann_func()
                        elif callable(ann):
                            ann = ann()
                    except Exception:  # pylint: disable=broad-except
                        ann = func_annotation.get(arg.arg, None)
                        if hasattr(ann, "type_ann_func"):
                            ann = ann.type_ann_func()
                        if ann is None:
                            raise
                    param = T.arg(arg.arg, ann)
                    self.var_table.add(arg.arg, param)
                self.visit_body(node.body)
    self.function_annotations = supplied_annotation


@dispatch.register(token="tir", type_name="tvm_annotation")
def visit_tvm_annotation(self: Parser, node: doc.expr):
    """The TVM annotation visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.expr
        The doc AST expr node.
    """
    annotation = self.eval_expr(node)
    if callable(annotation):
        annotation = annotation()
    return annotation


def _get_func_str(func):
    if isinstance(func, doc.Attribute):
        return f"{func.value.id}.{func.attr}"
    assert isinstance(func, doc.Name)
    return func.id


@dispatch.register(token="tir", type_name="Expr")
def visit_expr_stmt(self: Parser, node: doc.Expr) -> None:
    """The expr statement visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Expr
        The doc AST Expr node.
    """
    if CompassInfo.current() is not None:
        if isinstance(node.value, doc.Call):
            func_str = _get_func_str(node.value.func)
            if func_str in ("pdb.set_trace", "set_trace", "breakpoint"):
                return

    res = self.eval_expr(node.value)
    if res is None:
        pass
    elif isinstance(res, Frame):
        res.add_callback(partial(res.__exit__, None, None, None))
        res.__enter__()
    elif isinstance(res, PrimExpr):
        T.evaluate(res)
    elif isinstance(res, (int, bool)):
        T.evaluate(tvm.tir.const(res))
    elif isinstance(res, tvm.relax.Call) and not res.args:
        # Using GlobalVar.__call__ with no arguments is ambiguous, as
        # each IR has a different function Call representation.  If
        # this occurs, convert to the TIR representation.
        T.evaluate(tvm.tir.call_tir(res.op))
    elif isinstance(res, str):
        # Ignore docstrings
        pass
    elif isinstance(res, tvm.tir.stmt.BufferStore):
        T.buffer_store(res.buffer, res.value, res.indices, res.predicate)
    else:
        self.report_error(node, f"Parsing resulted in unexpected type {type(res)}")


@dispatch.register(token="tir", type_name="If")
def visit_if(self: Parser, node: doc.If) -> None:
    """The if visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.If
        The doc AST if node.
    """
    with self.var_table.with_frame():
        cond = self.eval_expr(node.test)
        cond = cond.asobject() if isinstance(cond, tvm.runtime.ObjectGeneric) else cond

        if not isinstance(cond, PrimExpr):
            with self.var_table.with_frame():
                if bool(cond) is True:
                    self.visit_body(node.body)
                elif node.orelse:
                    self.visit_body(node.orelse)
            return

        if cond.dtype.is_vector:
            self.report_error(node.test, "The condition required to be a scalar expression.")
        if cond.dtype == "handle":
            self.report_error(node.test, "Please check whether the pointer is null explicitly.")
        if cond.dtype != "bool":
            cond = tvm.tir._ffi_api._OpNE(cond, tvm.tir.const(0, cond.dtype), cond.span)

        with T.If(cond):
            with T.Then():
                with self.var_table.with_frame():
                    self.visit_body(node.body)
            if node.orelse:
                with T.Else():
                    with self.var_table.with_frame():
                        self.visit_body(node.orelse)


@dispatch.register(token="tir", type_name="Assert")
def visit_assert(self: Parser, node: doc.Assert) -> None:
    """The assert visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Assert
        The doc AST assert node.
    """
    cond = self.eval_expr(node.test)
    msg = self.eval_expr(node.msg)
    frame = T.Assert(cond, msg)
    frame.add_callback(partial(frame.__exit__, None, None, None))
    frame.__enter__()


@dispatch.register(token="tir", type_name="Return")
def visit_return(self: Parser, node: doc.Return) -> None:
    """The return visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Return
        The doc AST return node.
    """
    if node.value is not None:
        ret_value = self.eval_expr(node.value)

        if self.ret_type is None:
            self.report_error(node, "This function lacks return type annotation.")

        if isinstance(ret_value, Pointer):
            ret_type_scope = self.ret_type.storage_scope
            if ret_value.scope != ret_type_scope:
                param_scope = "private" if ret_type_scope == "local" else ret_type_scope
                arg_scope = "private" if ret_value.scope == "local" else ret_value.scope
                msg = f'The scope of return type expect "{param_scope}", but got: "{arg_scope}".'
                self.report_error(node, msg)

        T.evaluate(T.ret(ret_value))
        return
    T.evaluate(T.ret(None))


@dispatch.register(token="tir", type_name="Break")
def visit_break(self: Parser, node: doc.Break) -> None:  # pylint: disable=unused-argument
    """The break visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Break
        The doc AST break node.
    """
    T.evaluate(T.Break())


@dispatch.register(token="tir", type_name="tvm_declare_function")
def visit_tvm_declare_function(self: Parser, node: doc.FunctionDef) -> GlobalVar:
    """The function declaration step for tir

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Return
        The doc AST return node.
    """

    supplied_annotation = self.function_annotations
    func_annotation = supplied_annotation.get(node.name, {})

    ret_type = None
    with self.var_table.with_frame():
        if node.returns is not None:
            ret_type = self.eval_expr(node.returns)
            if callable(ret_type):
                ret_type = PrimType(ret_type().dtype)

        arg_annotations = []
        for arg in node.args.args:
            if arg.annotation is None:
                self.report_error(arg, "Type annotation required for function parameters.")
            try:
                ann = self.eval_expr(arg.annotation)
                if callable(ann):
                    ann = ann()
            except Exception:  # pylint: disable=broad-except
                ann = func_annotation.get(arg.arg, None)
                if ann is None:
                    raise

            IRBuilder.name(arg.arg, ann)
            arg_annotations.append(ann)

    func_signature = tvm.tir.PrimFunc(arg_annotations, None, ret_type=ret_type)
    return I.decl_function(node.name, func_signature)
