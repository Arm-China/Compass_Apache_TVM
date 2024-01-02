# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
"""Abstraction for pointer."""
from ..ir import PrimExpr, PointerType, PrimType
from ..runtime import DataType, ObjectGeneric
from .expr import Var
from .buffer import decl_buffer
from .op import pointer


class Pointer(ObjectGeneric):
    """Represent the concept corresponds to C/C++ pointer.

    The begin expression represents a base memory address, its data type can be
    different with the pointer. The offset expression represents a offset from
    the base memory address, in unit of data type of the pointer, the concrete
    address that the pointer represents is ((dtype*)begin + offset).
    """

    def __init__(self, dtype, scope, begin=None, offset=0, name=""):
        self.dtype = DataType(dtype)
        self.scope = scope
        self.begin = Var(name, PointerType(PrimType(dtype), scope)) if begin is None else begin
        self._offset = offset

        # The base memory address is a var or another pointer, the reason that it need to be another
        # pointer is representing the temporary pointer like '(pa + 3).as_ptr("fp16x16")' where "pa"
        # is a pointer of type "fp16", because the "3" in unit of "fp16" can't be converted to a
        # valid offset in unit of "fp16x16".
        assert isinstance(self.begin, (Var, Pointer)), f"Invalid begin type: {type(self.begin)}."
        self.buffer = None
        if isinstance(self.begin, Var) and not self.dtype.is_void:
            self.buffer = decl_buffer((-1,), dtype, f"{name}_buf", self.begin)

    def asobject(self):
        return pointer(self.dtype, self.scope, self.begin, self._offset)

    def as_ptr(self, dtype):
        assert dtype not in (None, ""), 'Please use "void" to indicate convert to void pointer.'
        if DataType(dtype) == self.dtype:
            return self
        # Only here may generate a pointer whose begin is another pointer.
        return Pointer(dtype, self.scope, self.begin if self._offset == 0 else self)

    def accessible_check(self):
        assert not self.dtype.is_void, "Can't access data through void pointer."
        assert isinstance(self.begin, Var), (
            "Can't access data through this temporary pointer, because it is converted from a "
            "different type and non-zero offset pointer, please define it as a new named pointer "
            "first and access data through the new named pointer."
        )

    def __getitem__(self, indices):
        self.accessible_check()
        assert not isinstance(indices, tuple), "Pointer only can be used to access 1D data."

        if isinstance(indices, slice):
            start = self._offset if indices.start is None else self._offset + indices.start
            assert indices.stop is not None, "The stop of the slice must be given."
            indices = slice(start, self._offset + indices.stop, indices.step)
        else:
            indices = self._offset + indices
        return self.buffer[indices]

    def _move_check(self, step):
        assert not self.dtype.is_void, "The void pointer can't be moved."
        if isinstance(step, int):
            return
        if isinstance(step, PrimExpr):
            dtype = DataType(step.dtype)
            if dtype.is_scalar and dtype.is_integer:
                return
        raise RuntimeError("The step that the pointer will be moved must be a scalar integer.")

    def __add__(self, other):
        self._move_check(other)
        return Pointer(self.dtype, self.scope, self.begin, self._offset + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        self._move_check(other)
        return Pointer(self.dtype, self.scope, self.begin, self._offset - other)
