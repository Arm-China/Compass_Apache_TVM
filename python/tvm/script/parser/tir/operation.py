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
"""The tir expression operation registration"""
#
# This file has been modified by Arm China team.
#

from typing import Type

from tvm import tir, target as tgt
from tvm._ffi.runtime_ctypes import DataType, int_within_range
from tvm.tir import IntImm

from .._core import OpMethod, doc, register_op


def _try_adjust_int_literal_type(lhs, rhs):
    # Only handle the situation that all operands are integers. For the situation that all operands
    # are floating, because can't know whether a float literal can be represented by float16 or not,
    # so can't do this. For other situations, C++ "BinaryOpMatchTypes" is good enough.
    if isinstance(lhs, int) and isinstance(rhs, tir.PrimExpr):
        rtype = DataType(rhs.dtype)
        if rtype.is_integer and int_within_range(lhs, rtype):
            return IntImm(rtype.element_of, lhs), rhs

    if isinstance(lhs, tir.PrimExpr) and isinstance(rhs, int):
        ltype = DataType(lhs.dtype)
        if ltype.is_integer and int_within_range(rhs, ltype):
            return lhs, IntImm(ltype.element_of, rhs)

    return lhs, rhs


def _gen_binary_wrapper(func):
    def _wrapper(a, b):
        a, b = _try_adjust_int_literal_type(a, b)
        return func(a, b)

    return _wrapper


def _register_expr_op(ty: Type):  # pylint: disable=invalid-name
    ty._dispatch_type = ty  # pylint: disable=protected-access

    def _and(a, b):
        if isinstance(a, bool):
            a = IntImm("bool", a)
        if isinstance(b, bool):
            b = IntImm("bool", b)
        if DataType(a.dtype).lanes > 1 or DataType(b.dtype).lanes > 1:
            if tgt.AipuInfo.current() is not None:
                err_msg = "Invalid and operator between vector, use & instead."
                raise TypeError(err_msg)
            return a & b
        else:
            return tir.And(a, b)

    def _or(a, b):
        if isinstance(a, bool):
            a = IntImm("bool", a)
        if isinstance(b, bool):
            b = IntImm("bool", b)
        if DataType(a.dtype).lanes > 1 or DataType(b.dtype).lanes > 1:
            if tgt.AipuInfo.current() is not None:
                err_msg = "Invalid or operator between vector, use | instead."
                raise TypeError(err_msg)
            return a | b
        else:
            return tir.Or(a, b)

    def r(op: Type, i: int, m: OpMethod):  # pylint: disable=invalid-name
        register_op(ty, op, i)(m)

    for i in [0, 1]:
        # Case 1. binop
        r(doc.Add, i, _gen_binary_wrapper(lambda a, b: a + b))
        r(doc.Sub, i, _gen_binary_wrapper(lambda a, b: a - b))
        r(doc.Mult, i, _gen_binary_wrapper(lambda a, b: a * b))
        r(doc.Div, i, _gen_binary_wrapper(lambda a, b: a / b))
        r(doc.FloorDiv, i, _gen_binary_wrapper(lambda a, b: a // b))
        r(doc.Mod, i, _gen_binary_wrapper(lambda a, b: a % b))
        r(doc.LShift, i, lambda a, b: a << b)
        r(doc.RShift, i, lambda a, b: a >> b)
        r(doc.BitOr, i, _gen_binary_wrapper(lambda a, b: a | b))
        r(doc.BitXor, i, _gen_binary_wrapper(lambda a, b: a ^ b))
        r(doc.BitAnd, i, _gen_binary_wrapper(lambda a, b: a & b))
        # doc.MatMult <-- not implemented
        r(doc.Pow, i, _gen_binary_wrapper(lambda a, b: a**b))
        # Case 2. cmpop
        r(doc.Eq, i, _gen_binary_wrapper(lambda a, b: tir._ffi_api._OpEQ(a, b, None)))
        r(doc.NotEq, i, _gen_binary_wrapper(lambda a, b: tir._ffi_api._OpNE(a, b, None)))
        r(doc.Lt, i, _gen_binary_wrapper(lambda a, b: a < b))
        r(doc.LtE, i, _gen_binary_wrapper(lambda a, b: a <= b))
        r(doc.Gt, i, _gen_binary_wrapper(lambda a, b: a > b))
        r(doc.GtE, i, _gen_binary_wrapper(lambda a, b: a >= b))
        # doc.Is <-- not implemented
        # doc.IsNot <-- not implemented
        # doc.In <-- not implemented
        # doc.NotIn <-- not implemented
        # Case 3. boolop
        r(doc.And, i, _and)
        r(doc.Or, i, _or)
    for i in [0]:
        #  Case 4. unaryop
        # doc.Invert <-- is overloaded
        r(doc.Not, i, tir.Not)
        # doc.UAdd <-- is overloaded
        # doc.USub <-- is overloaded


_register_expr_op(tir.PrimExpr)
_register_expr_op(tir.IterVar)
