# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Abstraction for pointer."""
from tvm import ir
from ..ir import PrimExpr, PointerType, PrimType, GlobalVar
from ..runtime import ObjectGeneric
from .expr import Var, Call, LT, LE, GT, GE, EqualOp, NotEqualOp
from .buffer import decl_buffer
from .op import pointer, isnullptr


def _in_block_check():
    from ..script.ir_builder import IRBuilder  # pylint: disable=import-outside-toplevel

    assert not IRBuilder.is_in_scope() or IRBuilder.current().find_frame("block") is None, (
        "Pointer can't be used in any schedulable block, please match it to a "
        "buffer and use the buffer instead."
    )


class Pointer(ObjectGeneric):
    """Represents the concept that corresponds to the C/C++ pointer.

    Just like the pointer of C/C++, below operations are supported.

    - Move forward and backward through adding or subtracting an integer value.
    - Read and write data as a 1-dimension array.
    - Check whether it is a null pointer or not.
    - Compare with the other pointer instance.
    - Cast to another type pointer.
    """

    def __init__(self, dtype, scope, base=None, offset=0, name=""):
        """The concrete address that the pointer represents is ((dtype*)base + offset).

        Parameters
        ----------
        dtype : DataType
            The data type of the data that the pointer point to.

        scope : str
            The memory space of the data that the pointer point to.

        base : Optional[Union[Var, Pointer, Call]]
            The base memory address, its data type can be different with the pointer.

        offset : Optional[Union[PrimExpr, int]
            The offset from the base memory address, in unit of data type of the pointer.

        name : Optional[str]
            The name of the pointer.
        """
        _in_block_check()
        self.dtype = dtype
        self.scope = scope
        self.base = Var(name, PointerType(PrimType(dtype), scope)) if base is None else base
        self.offset = offset

        # The "base" can be a PointerType var, a PrimType var, another pointer or a call that return
        # a pointer.
        # The reason that it need to be a PrimType var is representing the temporary pointer created
        # by getting address of a variable.
        # The reason that it need to be another pointer is representing the temporary pointer like
        # (pa + 3).as_ptr("fp16x16") where "pa" is a pointer of type "fp16", because the "3" in unit
        # of "fp16" can't be converted to a valid offset in unit of "fp16x16".
        # The reason that it need to be a call is representing the temporary pointer returned by
        # other functions, so the usage like "pa = func_return_ptr(x) + 2" can be supported.
        if isinstance(self.base, Call):
            if isinstance(self.base.op, GlobalVar):
                msg = f'Invalid base, call, return type: "{self.base.dtype}",'
                msg += f' op: "{self.base.op}".'
                assert self.base.dtype == "handle", msg
            else:
                msg = "Only accept reinterpret or GlobalVar when base is call,"
                msg += f' but got: "{self.base.op}".'
                assert self.base.op == ir.Op.get("tir.reinterpret"), msg
        else:
            assert isinstance(self.base, (Var, Pointer)), f"Invalid base type: {type(self.base)}."

        self.buffer = None
        if isinstance(self.base, Var) and self.base.dtype == "handle" and not self.dtype.is_void:
            self.buffer = decl_buffer((-1,), dtype, f"{name}_buf", self.base)

    @property
    def is_nullptr(self):
        """Check whether the current pointer instance is a null pointer or not."""
        _in_block_check()
        return isnullptr(self)

    def asobject(self):
        _in_block_check()
        return pointer(self.dtype, self.scope, self.base, self.offset)

    def as_ptr(self, dtype):
        """Cast to another pointer whose data type is the given one.

        Parameters
        ----------
        dtype : Union[str, DataType]
            The target data type.

        Returns
        -------
        ret : Pointer
            The new temporary pointer instance.
        """
        # pylint: disable=import-outside-toplevel
        from ..compass.dsl.utils import resolve_dtype_alias
        from ..compass.dsl.utils import VALID_PTR_ELEMENT_DTYPES

        assert dtype not in (None, ""), 'Please use "void" to indicate convert to void pointer.'
        dtype = resolve_dtype_alias(dtype)
        msg = f'The scalar form of arg "dtype" expect one of {VALID_PTR_ELEMENT_DTYPES}, but '
        msg += f'got: "{dtype.element_of}".'
        assert dtype.element_of in VALID_PTR_ELEMENT_DTYPES, msg

        if dtype == self.dtype:
            return self
        # Only here may generate a pointer whose base is another pointer.
        base = self.base if not isinstance(self.base, Call) and self.offset == 0 else self
        return Pointer(dtype, self.scope, base)

    def accessible_check(self, indices):
        """Applied when accessing data, check and report errors."""
        _in_block_check()
        assert not self.dtype.is_void, "Can't access data through void pointer."
        assert not (isinstance(self.base, Var) and self.base.dtype != "handle"), (
            "Can't access data through this temporary pointer, because it is created by getting "
            "address of a variable in current scope, please access data through the variable "
            "directly."
        )
        assert not isinstance(self.base, Call), (
            "Can't access data through this temporary pointer, because it is returned by other "
            "functions, please define it as a new named pointer first and access data through the "
            "new named pointer."
        )
        assert not isinstance(self.base, Pointer), (
            "Can't access data through this temporary pointer, because it is converted from a "
            "different type and non-zero offset pointer, please define it as a new named pointer "
            "first and access data through the new named pointer."
        )
        assert not isinstance(indices, tuple), "Pointer only can be used to access 1D data."
        msg = "For accessing data through pointer, the stop of the slice must be given."
        assert not (isinstance(indices, slice) and indices.stop is None), msg

    def __getitem__(self, indices):
        """Read data as a 1-dimension array.

        Parameters
        ----------
        indices : Union[sgentype, slice]
            The index used to access the concrete data. Multiple data will be read if it is a slice.

        Returns
        -------
        ret : single or multiple times of the data type of the pointer
            The result data that is read out.
        """
        self.accessible_check(indices)

        if isinstance(indices, slice):
            start = self.offset if indices.start is None else self.offset + indices.start
            indices = slice(start, self.offset + indices.stop, indices.step)
        else:
            indices = self.offset + indices
        return self.buffer[indices]

    def _move_check(self, step):
        assert not self.dtype.is_void, "The void pointer can't be moved."
        assert isinstance(step, int) or (
            isinstance(step, PrimExpr) and step.dtype.is_integer_scalar
        ), "The step that the pointer will be moved must be a scalar integer."

    def __add__(self, other):
        """Move the pointer to the higher address space.

        Parameters
        ----------
        other : sgentype
            The step that the pointer will be moved, in units of data type of the pointer.

        Returns
        -------
        ret : Pointer
            The new temporary pointer instance.
        """
        self._move_check(other)
        return Pointer(self.dtype, self.scope, self.base, self.offset + other)

    def __radd__(self, other):
        """Move the pointer to the higher address space.

        Parameters
        ----------
        other : sgentype
            The step that the pointer will be moved, in units of data type of the pointer.

        Returns
        -------
        ret : Pointer
            The new temporary pointer instance.
        """
        return self.__add__(other)

    def __sub__(self, other):
        """Move the pointer to the lower address space.

        Parameters
        ----------
        other : sgentype
            The step that the pointer will be moved, in units of data type of the pointer.

        Returns
        -------
        ret : Pointer
            The new temporary pointer instance.
        """
        self._move_check(other)
        return Pointer(self.dtype, self.scope, self.base, self.offset - other)

    def _comparable_check(self, other):
        assert isinstance(other, Pointer), "Only can compare pointer with another pointer."

    def __lt__(self, other):
        """Check whether the address that the pointer represents is < that of "other".

        Parameters
        ----------
        other : Pointer
            The other pointer instance that will be compared with.

        Returns
        -------
        ret : bool
            The compare result.
        """
        self._comparable_check(other)
        return LT(self, other)

    def __le__(self, other):
        """Check whether the address that the pointer represents is <= that of "other".

        Parameters
        ----------
        other : Pointer
            The other pointer instance that will be compared with.

        Returns
        -------
        ret : bool
            The compare result.
        """
        self._comparable_check(other)
        return LE(self, other)

    def __gt__(self, other):
        """Check whether the address that the pointer represents is > that of "other".

        Parameters
        ----------
        other : Pointer
            The other pointer instance that will be compared with.

        Returns
        -------
        ret : bool
            The compare result.
        """
        self._comparable_check(other)
        return GT(self, other)

    def __ge__(self, other):
        """Check whether the address that the pointer represents is >= that of "other".

        Parameters
        ----------
        other : Pointer
            The other pointer instance that will be compared with.

        Returns
        -------
        ret : bool
            The compare result.
        """
        self._comparable_check(other)
        return GE(self, other)

    def same_as(self, other):
        # Will be called by EqualOp or NotEqualOp, when the compare isn't happened inside TVM script
        # program.
        return super().__eq__(other)

    def __eq__(self, other):
        """Check whether the address that the pointer represents is == that of "other".

        Parameters
        ----------
        other : Pointer
            The other pointer instance that will be compared with.

        Returns
        -------
        ret : bool
            The compare result.
        """
        self._comparable_check(other)
        return EqualOp(self, other)

    def __ne__(self, other):
        """Check whether the address that the pointer represents is != that of "other".

        Parameters
        ----------
        other : Pointer
            The other pointer instance that will be compared with.

        Returns
        -------
        ret : bool
            The compare result.
        """
        self._comparable_check(other)
        return NotEqualOp(self, other)
