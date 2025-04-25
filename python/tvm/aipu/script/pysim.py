# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""The pysim part of Zhouyi Compass script APIs."""
import os
import time
import uuid
import threading
import concurrent
import numpy as np
from tvm import ir, tir, script, DataType, get_range
from ..logger import WARN
from ..utils import get_binary_op_result_type, resolve_dtype_alias, control_option, abspath
from ..tir.aiff import DescChainArray


_RANDOM_ARRAY = np.random.random((64,)).astype("float32")


def _is_in_ir_api():
    return PySimInfo.current.thread_local_data.is_in_ir_api


def binary_op_match_types(lhs, rhs):
    """Ensure the data type of lhs and rhs are matched."""
    from .ir.conversion import cast  # pylint: disable=import-outside-toplevel

    ltype_or_lhs = lhs.dtype if isinstance(lhs, (PyVar, tir.PrimExpr)) else lhs
    rtype_or_rhs = rhs.dtype if isinstance(rhs, (PyVar, tir.PrimExpr)) else rhs
    ret_dtype = get_binary_op_result_type(ltype_or_lhs, rtype_or_rhs)

    if PySimInfo.current is None:
        return cast(lhs, ret_dtype), cast(rhs, ret_dtype)

    # Implementation of PySim.
    return np.asarray(lhs, dtype=ret_dtype), np.asarray(rhs, dtype=ret_dtype)


class PyVar:
    """The Python implementation of class "tir.Var"."""

    def __init__(self, value, dtype=None, mask=None, r=None):
        if dtype is None:
            self.dtype = DataType(value.dtype)
            if value.ndim != 0:
                self.dtype = self.dtype.with_lanes(len(value))
        else:
            self.dtype = DataType(dtype)
        self.value = np.array(value, dtype=self.dtype.element_of)

        if mask is None or all(mask):
            return
        # Fill value of the inactive elements according to the predication mask.
        r = _RANDOM_ARRAY.view(self.dtype.element_of)[: self.dtype.lanes] if r is None else r
        self.value = np.where(mask, self.value, r)

    @classmethod
    def zeros(cls, dtype, *args, **kwargs):
        return cls([0] * dtype.lanes, dtype, *args, **kwargs)

    def astype(self, dtype):
        return self.value.astype(dtype)

    def view(self, dtype):
        return self.value.view(dtype)

    def copy(self):
        return PyVar(self.value, self.dtype)

    @property
    def addr(self):
        return PyPointer(self.dtype, "local", self.value.reshape(-1).view("uint8"))

    def __array__(self, dtype=None, copy=None):  # pylint: disable=unused-argument
        return self.value

    def __getitem__(self, idx):
        ret = self.value[idx]
        if _is_in_ir_api():
            return ret

        # Call from the DSL program directly instead of the PySim version of IR
        # APIs, so the result need to be a PyVar instance.
        assert idx >= 0, "The index must >= 0 when access element of vector variable."

        if self.dtype.is_bool:  # Scalar bool needn't to be represented by PyVar.
            return bool(ret)
        return PyVar(ret, self.dtype.with_lanes(1))

    def __setitem__(self, idx, value):
        value = value.value if isinstance(value, PyVar) else value
        if _is_in_ir_api():
            self.value[idx] = value
            return

        assert idx >= 0, "The index must >= 0 when access element of vector variable."
        self.value[idx] = value

    def __index__(self):
        return int(self.value)

    def __add__(self, other):
        if isinstance(other, PyPointer):
            return other.__radd__(self)

        if _is_in_ir_api():
            return self.value + other

        lhs, rhs = binary_op_match_types(self, other)
        return PyVar(lhs + rhs)

    def __mul__(self, other):
        if isinstance(other, (tuple, list)):
            return int(self.value) * other

        if _is_in_ir_api():
            return self.value * other

        lhs, rhs = binary_op_match_types(self, other)
        return PyVar(lhs * rhs)

    def __rmul__(self, other):
        return self.__mul__(other)

    def _div(self, lhs, rhs):
        lhs, rhs = binary_op_match_types(lhs, rhs)
        ret = lhs / rhs

        if DataType(lhs.dtype).is_integer:  # Indicate it's a integer division.
            if ret.ndim != 0:  # Indicate it's a integer vector division.
                # Align with AIPU, integer vector division will do saturation, because integer
                # division won't underoverflow, so only need care the upper bound.
                ret = np.minimum(ret, get_range(lhs.dtype)[1])
            # Align with C, so here actually is truncated division.
            ret = ret.astype(lhs.dtype)
        return PyVar(ret)

    def __truediv__(self, other):
        return self.value / other if _is_in_ir_api() else self._div(self, other)

    def __rtruediv__(self, other):
        return other / self.value if _is_in_ir_api() else self._div(other, self)

    def __floordiv__(self, other):
        # All operands are guaranteed to be integer in compilation stage.
        return self.value // other if _is_in_ir_api() else self._div(self, other)

    def __rfloordiv__(self, other):
        return other // self.value if _is_in_ir_api() else self._div(other, self)

    def _mod(self, lhs, rhs):
        # Align with C, so here actually is truncated modulo, and all operands
        # are guaranteed to be integer in compilation stage.
        lhs, rhs = binary_op_match_types(lhs, rhs)
        return PyVar(np.fmod(lhs, rhs))

    def __mod__(self, other):
        return self.value % other if _is_in_ir_api() else self._mod(self, other)

    def __rmod__(self, other):
        return other % self.value if _is_in_ir_api() else self._mod(other, self)

    def __repr__(self):
        return f"PyVar({self.dtype}) {self.value}"

    @property
    def hex(self):
        value = self.value.view(f"uint{self.dtype.bits}")
        return hex(value) if self.dtype.is_scalar else tuple(hex(x) for x in value)


def _gen_unary_method(np_func):
    def _method(self):
        ret = np_func(self.value)
        return ret if _is_in_ir_api() else PyVar(ret, self.dtype)

    return _method


def _gen_binary_method(np_func):
    def _method(self, other):
        if _is_in_ir_api():
            return np_func(self.value, other)

        lhs, rhs = binary_op_match_types(self, other)
        return PyVar(np_func(lhs, rhs))

    return _method


def _shift_op_match_types(lhs, rhs):
    # All operands are guaranteed to be integer in compilation stage.
    assert all(isinstance(x, (int, PyVar)) for x in (lhs, rhs))
    lhs = np.array(lhs, dtype="int32") if isinstance(lhs, int) else lhs.value
    rhs = np.array(rhs, dtype="int32") if isinstance(rhs, int) else rhs.value

    if lhs.dtype == rhs.dtype:
        return lhs, rhs

    return lhs, rhs.astype(lhs.dtype)


def _gen_shift_method(np_func):
    def _method(self, other):
        if _is_in_ir_api():
            return np_func(self.value, other)

        lhs, rhs = _shift_op_match_types(self, other)

        # Align with AIPU, for the scalar situation, the shift value need to be modulo 32.
        func = (
            lambda x: np.array(x % 32, dtype=x.dtype) if self.dtype.is_scalar and x.ndim == 0 else x
        )
        if np_func.__name__ in ("__rlshift__", "__rrshift__"):
            lhs = func(lhs)
        else:
            rhs = func(rhs)
        return PyVar(np_func(lhs, rhs))

    return _method


def _gen_compare_method(np_func):
    def _method(self, other):
        if _is_in_ir_api():
            return np_func(self.value, other)

        lhs, rhs = binary_op_match_types(self, other)
        ret = np_func(lhs, rhs)

        if ret.ndim == 0:  # Scalar bool needn't to be represented by PyVar.
            return bool(ret)
        return PyVar(ret)

    return _method


_unary_methods = ("__neg__", "__invert__")

_binary_methods = ("__radd__", "__sub__", "__rsub__", "__pow__", "__rpow__")
_binary_methods += ("__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__")

_shift_methods = ("__lshift__", "__rlshift__", "__rshift__", "__rrshift__")
_compare_methods = ("__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__")

for _name in _unary_methods:
    setattr(PyVar, _name, _gen_unary_method(getattr(np.ndarray, _name)))

for _name in _binary_methods:
    setattr(PyVar, _name, _gen_binary_method(getattr(np.ndarray, _name)))

for _name in _shift_methods:
    setattr(PyVar, _name, _gen_shift_method(getattr(np.ndarray, _name)))

for _name in _compare_methods:
    setattr(PyVar, _name, _gen_compare_method(getattr(np.ndarray, _name)))


def _get_flatten_offset(shape, indices):
    assert len(indices) <= len(shape)
    ret = 0
    for i, idx in enumerate(indices):
        ret += idx * int(np.prod(shape[i + 1 :]))
    return ret


class PyBuffer:
    """The Python implementation of class "tir.Buffer"."""

    def __init__(self, dtype, shape, scope, u8_np_arr, u8_offset=0):
        self.dtype = DataType(dtype)
        self._scope = scope
        assert len(u8_np_arr.shape) == 1 and u8_np_arr.dtype == "uint8"
        self._u8_np_arr = u8_np_arr
        self._u8_offset = u8_offset

        shape = (shape,) if not isinstance(shape, (list, tuple)) else shape
        self.shape = tuple(int(x) for x in shape)
        assert all(x > 0 for x in self.shape), f'Shape: "{self.shape}" can\'t have negative value.'
        np_shape = self.shape + (self.dtype.lanes,) if self.dtype.is_vector else self.shape
        end_offset = u8_offset + np.prod(np_shape) * self.dtype.bytes

        missing_bytes = end_offset - u8_np_arr.size
        msg = f'Need more "{missing_bytes}" bytes data to match the buffer with dtype: '
        msg += f'"{self.dtype}" and shape: "{self.shape}".'
        assert missing_bytes <= 0, msg

        self._np_arr = u8_np_arr[u8_offset:end_offset].view(self.dtype.element_of).reshape(np_shape)

    def __getitem__(self, indices):
        ret = self._np_arr[indices]
        ret = ret.reshape(-1) if self.dtype.is_vector else ret  # Keep the data as 1D.

        if _is_in_ir_api():
            return ret
        return PyVar(ret)

    def __setitem__(self, indices, value):
        value = value.value if isinstance(value, PyVar) else value

        if self.dtype.is_vector:
            self._np_arr[indices].reshape(-1)[:] = value  # Treat the data will be set as 1D.
        else:
            self._np_arr[indices] = value

    def addr_of(self, indices):
        """Obtain a pointer that point to the address of the element on the given indices.

        Parameters
        ----------
        indices : Union[int, Tuple[int], List[int]]
            The indices of the element in the buffer.

        Returns
        -------
        ret: PyPointer
            The result pointer instance.
        """
        indices = tuple(indices) if isinstance(indices, (tuple, list)) else (indices,)
        indices = indices + (0,) * (len(self.shape) - len(indices))
        new_offset = _get_flatten_offset(self.shape, indices)
        new_offset = self._u8_offset + new_offset * self.dtype.total_bytes
        return PyPointer(self.dtype, self._scope, self._u8_np_arr, new_offset)

    def __repr__(self):
        return f"PyBuffer({self.dtype}, {self.shape}, {self._scope}) {self._np_arr}"


class PyDescPointer:
    """The Python implementation of descriptor pointer."""

    def __init__(self, dtype, scope, chains, offset=0):
        self._dtype = DataType(dtype)
        self._scope = scope
        self._chains = chains
        self._offset = offset
        self._count_of_u32 = sum(x.count_of_u32 for x in self._chains)
        self.is_nullptr = False

    def get_current_desc_chain(self):
        """Get the descriptor chain pointed by the current pointer."""
        cur_offset = self._offset

        for chain in self._chains:
            assert cur_offset >= 0, "The offset isn't on the bound of any descriptor chain."
            if cur_offset == 0:
                return DescChainArray((chain,))

            cur_offset -= chain.count_of_u32

        raise IndexError("The offset is out of range.")

    def _check_index(self, index):
        assert isinstance(index, int), "Currently can't support slice."
        assert index + self._offset < self._count_of_u32, "The index is out of range."
        assert index >= 0, "The index must >= 0 when access data through pointer."

    def __getitem__(self, idx):
        self._check_index(idx)

        ret = None
        idx += self._offset
        for chain in self._chains:
            if idx < chain.count_of_u32:
                ret = chain.get_item_as_flatten_list(idx)
                break
            idx -= chain.count_of_u32

        if _is_in_ir_api():
            return ret

        if isinstance(ret, np.ndarray):
            return PyPointer(ret.dtype, "global", ret.reshape(-1).view("uint8"))

        return PyVar(ret, dtype=self._dtype)

    def __setitem__(self, idx, value):
        self._check_index(idx)
        value = value.value if isinstance(value, PyVar) else value

        idx += self._offset
        for chain in self._chains:
            if idx < chain.count_of_u32:
                if isinstance(value, PyPointer):
                    value = value.np_arr
                chain.set_item_as_flatten_list(idx, value)
                break
            idx -= chain.count_of_u32

    def __add__(self, other):
        new_offset = self._offset + int(other)
        return PyDescPointer(self._dtype, self._scope, self._chains, new_offset)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        new_offset = self._offset - int(other)
        return PyDescPointer(self._dtype, self._scope, self._chains, new_offset)

    def __repr__(self):
        return f"PyDescPointer({self._dtype}, {self._scope}) {self._chains}"


class PyPointer:
    """The Python implementation of class "tir.Pointer"."""

    def __init__(self, dtype, scope, u8_np_arr, u8_offset=0):
        self.dtype = DataType(dtype)
        self.scope = scope
        assert u8_np_arr is None or (len(u8_np_arr.shape) == 1 and u8_np_arr.dtype == "uint8")
        self.u8_np_arr = u8_np_arr
        self.u8_offset = u8_offset
        self.is_nullptr = u8_np_arr is None
        self.np_arr = None

        if self.dtype.is_void or self.is_nullptr:
            return

        cnt = (u8_np_arr.size - u8_offset) // self.dtype.total_bytes
        if cnt <= 0:
            return

        end_offset = u8_offset + cnt * self.dtype.total_bytes
        shape = (-1, self.dtype.lanes) if self.dtype.is_vector else (-1,)
        self.np_arr = u8_np_arr[u8_offset:end_offset].view(self.dtype.element_of).reshape(shape)

    def as_ptr(self, dtype):
        dtype = resolve_dtype_alias(dtype)
        if dtype == self.dtype:
            return self
        return PyPointer(dtype, self.scope, self.u8_np_arr, self.u8_offset)

    def _check_index(self, index):
        assert self.np_arr is not None, "Out of bounds."

        if _is_in_ir_api():
            return

        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            step = 1 if index.step is None else index.step
            msg = "The stop of the slice must > the start of the slice >= 0."
            assert index.stop > start >= 0, msg
            assert step > 0, "The step of the slice must > 0 when access data through pointer."
        else:
            assert index >= 0, "The index must >= 0 when access data through pointer."

    def __getitem__(self, indices):
        self._check_index(indices)

        ret = self.np_arr[indices]
        ret = ret.reshape(-1) if self.dtype.is_vector else ret  # Keep the data as 1D.

        if _is_in_ir_api():
            return ret
        return PyVar(ret)

    def __setitem__(self, indices, value):
        self._check_index(indices)
        value = value.value if isinstance(value, PyVar) else value

        if self.dtype.is_vector:
            self.np_arr[indices].reshape(-1)[:] = value  # Treat the data will be set as 1D.
        else:
            self.np_arr[indices] = value

    def __add__(self, other):
        new_offset_in_byte = self.u8_offset + int(other) * self.dtype.total_bytes
        return PyPointer(self.dtype, self.scope, self.u8_np_arr, new_offset_in_byte)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        new_offset_in_byte = self.u8_offset - int(other) * self.dtype.total_bytes
        return PyPointer(self.dtype, self.scope, self.u8_np_arr, new_offset_in_byte)

    def __repr__(self):
        return f"PyPointer({self.dtype}, {self.scope}) {self.np_arr}"

    def _comparable_check(self, other):
        assert self.u8_np_arr is other.u8_np_arr, "Can't compare two irrelevant pointers."

    def __lt__(self, other):
        self._comparable_check(other)
        return self.u8_offset < other.u8_offset

    def __le__(self, other):
        self._comparable_check(other)
        return self.u8_offset <= other.u8_offset

    def __gt__(self, other):
        self._comparable_check(other)
        return self.u8_offset > other.u8_offset

    def __ge__(self, other):
        self._comparable_check(other)
        return self.u8_offset >= other.u8_offset

    def __eq__(self, other):
        self._comparable_check(other)
        return self.u8_offset == other.u8_offset

    def __ne__(self, other):
        self._comparable_check(other)
        return self.u8_offset != other.u8_offset


class PyEvent:
    """The Python implementation of Zhouyi NPU event."""

    def __init__(self, idx):
        self._idx = idx
        self.is_free = True
        self._dependency_chain = []

    def occupy_as_producer(self, idx):
        if idx == 0:
            return
        self._dependency_chain[idx - 1].wait()

    def increase_producer(self):
        ret = len(self._dependency_chain)
        self._dependency_chain.append(threading.Event())
        return ret

    def trigger_as_producer(self, idx):
        self._dependency_chain[idx].set()

    def wait(self):
        if len(self._dependency_chain) == 0:
            return
        self._dependency_chain[-1].wait()

        assert all(x.is_set() for x in self._dependency_chain)
        self._dependency_chain.clear()

    def reset(self):
        self.is_free = True
        self._dependency_chain.clear()


def random_pause():
    if not control_option.is_random_pause:
        return

    tec_count = PySimInfo.current.local_size
    tid = PySimInfo.current.thread_local_data.id
    # First wait to ensure the order of getting random sequence.
    time.sleep(tid * 0.1)
    time.sleep(np.random.choice(range(tec_count)))


class PySimInfo:
    """Maintain all of the status information when simulate the TVM script in Python."""

    current = None

    def __init__(self, aipu_info, output_dir, is_multi_thread=True):
        self.aipu_info = aipu_info
        self.local_size = 4
        self.thread_local_data = threading.local()
        self.barrier = threading.Barrier(self.local_size)
        self.cur_shared_buffer = None
        self.is_multi_thread = is_multi_thread
        self.output_dir = f"{output_dir}/runtime/pysim"
        self._old_current = None

    def __enter__(self):
        self._old_current, PySimInfo.current = PySimInfo.current, self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        PySimInfo.current = self._old_current


class PyPrimFunc:
    """The Python implementation of class "tir.PrimFunc"."""

    def __init__(self, py_func, attrs):
        self.py_func = py_func
        self.attrs = attrs
        self.param_infos = None
        self.aipu_info = None
        self.output_dir = None
        self._param_anns = []
        self._param_names = []
        self._return_ann = None

        for name, ann in py_func.__annotations__.items():
            if hasattr(ann, "type_ann_func"):
                ann = ann.type_ann_func()
            else:
                msg = "The type annotation of "
                msg += "return value" if name == "return" else f'parameter "{name}"'
                msg += f' of function "{py_func.__name__}" is not supported.'
                assert isinstance(ann, tir.Pointer), msg

            if name == "return":
                self._return_ann = ann
            else:
                self._param_anns.append(ann)
                self._param_names.append(name)

    def _convert_sub_func_arg(self, idx, arg):
        param = self._param_anns[idx]

        # 1. Arguments annotated as variable, by something like "S.i32", "S.i32x8".
        if isinstance(param, tir.Var):
            return PyVar(arg, param.dtype)

        # 2. Arguments annotated by "S.ptr".
        # 2.1 The actual argument is descriptor.
        if isinstance(arg, PyDescPointer):
            return arg

        # 2.2 The actual argument is normal data.
        assert isinstance(arg, (PyPointer, PyBuffer))
        arg = arg.addr_of(0) if isinstance(arg, PyBuffer) else arg
        return arg.as_ptr(param.dtype)

    def _convert_entry_func_arg(self, idx, arg):
        param = self._param_anns[idx]

        # 1. Arguments annotated as scalar variable, by something like "S.i32", "S.size_i32".
        if isinstance(param, tir.Var):
            msg = "A vector variable can't be a argument of the entry function."
            assert DataType(param.dtype).is_scalar, msg
            return PyVar(arg, param.dtype)

        # 2. Arguments annotated by "S.ptr".
        # 2.1 The actual argument is descriptor.
        if isinstance(arg, DescChainArray):
            # TODO: Add type check to ensure it is not a empty DescChainArray.
            return PyDescPointer(param.dtype, param.scope, tuple(x for x in arg))

        # 2.2 The actual argument is normal data.
        if isinstance(arg, np.ndarray):
            param_elem_dtype = param.dtype.element_of
            if param_elem_dtype != "void" and arg.dtype != param_elem_dtype:
                msg = f'The {idx + 1}-th arg of function "{self.__name__}" expect a '
                msg += f'{param_elem_dtype} NumPy, but got: "{arg.dtype}".'
                WARN(msg)

            # Ensure array is C-style contiguous, otherwise the view action will get wrong result.
            msg = f'The {idx + 1}-th arg of function "{self.__name__}" expect a C-style contiguous '
            msg += 'NumPy, please achieve it through "numpy.ascontiguousarray".'
            assert arg.flags.c_contiguous, msg
            u8_np_arr = arg.view("uint8").reshape(-1)
        else:
            msg = f'The {idx + 1}-th arg expect a NumPy, but got: "{type(arg)}".'
            assert arg in (None, 0), msg  # Indicate it is a null pointer.
            u8_np_arr = None

        return PyPointer(param.dtype, param.scope, u8_np_arr)

    def _implicit_convert_with_check(self, args):
        from .ir.utils import is_scalar  # pylint: disable=import-outside-toplevel

        param_cnt, arg_cnt = len(self._param_anns), len(args)
        msg = f'The function "{self.__name__}" expect {param_cnt} args, but got: "{arg_cnt}".'
        assert arg_cnt == param_cnt, msg

        ret = []
        for i, arg in enumerate(args):
            param = self._param_anns[i]
            arg = arg.addr_of(0) if isinstance(arg, tir.Buffer) else arg

            # 1. Arguments annotated as variable, by something like "S.i32", "S.i32x8".
            if isinstance(param, tir.Var):
                # 1.1 Arguments annotated as variable but got pointer.
                msg = f"The {i + 1}-th arg expect a variable, but got: pointer."
                assert not isinstance(arg, tir.Pointer), msg

                # 1.2 Arguments annotated as scalar variable.
                if DataType(param.dtype).is_scalar:
                    assert is_scalar(arg), f"The {i + 1}-th arg expect a scalar."
                else:
                    # 1.3 Arguments annotated as vector variable.
                    msg = f"The {i + 1}-th arg expect a vector."
                    assert isinstance(arg, tir.PrimExpr) and DataType(arg.dtype).is_vector, msg

                ret.append(arg)
                continue

            # 2. Arguments annotated as pointer, by something like "S.ptr".
            assert isinstance(arg, tir.Pointer), f"The {i + 1}-th arg expect a pointer."

            param_scope = "private" if param.scope == "local" else param.scope
            arg_scope = "private" if arg.scope == "local" else arg.scope
            msg = f'The scope of {i + 1}-th arg expect "{param_scope}", but got: "{arg_scope}".'
            assert arg.scope == param.scope, msg

            ret.append(arg)

        return ret

    def _append_kwargs_to_args(self, args, kwargs):
        func_name = self.__name__
        defaults = self.py_func.__defaults__ or []
        default_names = self._param_names[len(self._param_anns) - len(defaults) :]
        dft_dict = dict(zip(default_names, defaults))

        # Check args number.
        arg_cnt, kwarg_cnt = len(args), len(kwargs)
        param_cnt, default_cnt = len(self._param_names), len(dft_dict)
        arg_kwarg_cnt = arg_cnt + kwarg_cnt
        start = param_cnt - default_cnt
        if not (start <= arg_kwarg_cnt <= param_cnt):
            start = param_cnt - default_cnt
            expect = f"{start} to {param_cnt}" if start != param_cnt else f"{param_cnt}"
            msg = f'The function "{func_name}" expect {expect} args,'
            msg += f' but got: "{arg_kwarg_cnt}".'
            raise TypeError(msg)

        # Check kwargs are valid.
        for karg in kwargs.keys():
            msg = f'The function "{func_name}" got unexpect keyword args: "{karg}".'
            assert karg in self._param_names, msg

        # Check overlapping for args and kwargs.
        arg_dict = dict(zip(self._param_names, args))
        keys_intersect = ", ".join(arg_dict.keys() & kwargs.keys())
        if len(keys_intersect) > 0:
            msg = f'The function "{func_name}" got multiple values for args: "{keys_intersect}".'
            raise TypeError(msg)

        # Construct result starting from default.
        dft_dict.update(arg_dict)
        dft_dict.update(kwargs)

        # Check args number.
        arg_cnt = len(dft_dict)
        if arg_cnt < param_cnt:
            msg = f'The function "{func_name}" missing "{param_cnt - arg_cnt}"'
            msg += f' args: "{",".join(self._param_names - dft_dict.keys())}".'
            raise TypeError(msg)
        if arg_cnt > param_cnt:
            expect = f"{start} to {param_cnt}" if start != param_cnt else f"{param_cnt}"
            msg = f'The function "{func_name}" expect {expect} args, but got: "{arg_cnt}".'
            raise TypeError(msg)

        # Extract args based on the params order defined in the function.
        return tuple(dft_dict[k] for k in self._param_names)

    def __call__(self, *args, **kwargs):
        if script.ir_builder.IRBuilder.is_in_scope():
            # It is evaluating by the TVM script parser when parsing other functions who has called
            # this function, so just need to return a call node to this function or a pointer.
            args = self._append_kwargs_to_args(args, kwargs)

            ret_ann = self._return_ann
            call_dtype = "void"
            if isinstance(ret_ann, tir.Var):
                call_dtype = ret_ann.dtype
            elif isinstance(ret_ann, tir.Pointer):
                call_dtype = "handle"

            args = self._implicit_convert_with_check(args)
            call = tir.Call(call_dtype, ir.GlobalVar(self.__name__), args)
            if isinstance(ret_ann, tir.Pointer):
                return tir.Pointer(ret_ann.dtype, ret_ann.scope, call)
            return call

        if PySimInfo.current is not None:  # Indicate the current instance isn't a entry function.
            args = [self._convert_sub_func_arg(i, x) for i, x in enumerate(args)]
            if len(kwargs) > 0:
                kwargs = {
                    k: self._convert_sub_func_arg(self._param_names.index(k), v)
                    for k, v in kwargs.items()
                }
            return self.py_func(*args, **kwargs)

        # Indicate the current instance is an entry function, so need to check and setup the
        # simulation environment.
        assert self.param_infos is not None, "Please compile before execute by Python Simulation."
        param_cnt, arg_cnt = len(self._param_anns), len(args)
        msg = f'The function "{self.__name__}" expect {param_cnt} args, but got: "{arg_cnt}".'
        assert arg_cnt == param_cnt, msg

        args = tuple(self._convert_entry_func_arg(i, x) for i, x in enumerate(args))

        if os.getenv("AIPU_TVM_PYSIM_SINGLE_THREAD", "FALSE").upper() == "TRUE":
            WARN("PySim is running in single thread, some data race issues may can't be caught.")
            with PySimInfo(self.aipu_info, self.output_dir, is_multi_thread=False) as py_sim_info:
                for i in range(py_sim_info.local_size):
                    py_sim_info.thread_local_data.id = i
                    py_sim_info.thread_local_data.is_in_ir_api = False
                    py_sim_info.thread_local_data.events = tuple(PyEvent(x) for x in range(4))

                    self.py_func(*[x.copy() if isinstance(x, PyVar) else x for x in args])

            return None

        with PySimInfo(self.aipu_info, self.output_dir, is_multi_thread=True) as py_sim_info:

            def _run(future, thread_id):
                py_sim_info.thread_local_data.id = thread_id
                py_sim_info.thread_local_data.is_in_ir_api = False
                py_sim_info.thread_local_data.events = tuple(PyEvent(x) for x in range(4))

                random_pause()

                try:
                    self.py_func(*[x.copy() if isinstance(x, PyVar) else x for x in args])
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(None)

            futures = []
            for i in range(py_sim_info.local_size):
                future = concurrent.futures.Future()
                threading.Thread(target=_run, name=f"TEC{i}", args=(future, i)).start()
                futures.append(future)

            for future in futures:
                # The exceptions raised in the sub-thread will be re-raised
                # here, so the main thread can catch and handle them.
                future.result()

        return None


_RUN_SIM = None  # To solve the circular import error.


def _set_run_sim(x):
    global _RUN_SIM
    _RUN_SIM = x


def pysim_run_sim(code_snippet, inputs=None, outputs=None, consts=None, descs=None):
    """Simple run the specified code snippet through AIPU simulator during PySim running."""
    inputs = tuple() if inputs is None else inputs
    outputs = tuple() if outputs is None else outputs
    consts = tuple() if consts is None else consts

    pysim_info = PySimInfo.current
    output_dir = abspath(f"{pysim_info.output_dir}/run_sim_{uuid.uuid4().hex}")
    inputs = tuple((name, x.np_arr if isinstance(x, PyPointer) else x) for name, x in inputs)
    outputs = tuple((name, x.np_arr if isinstance(x, PyPointer) else x) for name, x in outputs)
    consts = tuple((name, x.np_arr if isinstance(x, PyPointer) else x) for name, x in consts)
    _RUN_SIM(pysim_info.aipu_info, output_dir, code_snippet, inputs, outputs, consts, descs)
