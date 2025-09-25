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
#
# This file has been modified by Arm China team.
#
"""TVM runtime namespace."""

# class exposures
from .packed_func import PackedFunc
from .object import Object
from .object_path import ObjectPath, ObjectPathPair
from .script_printer import Scriptable
from .object_generic import ObjectGeneric
from .device import Device
from .ndarray import NDArray
from .module import Module
from .profiling import Report
from .executable import Executable

# function exposures
from .ndarray import device, cpu, cuda, opencl, vulkan, metal
from .ndarray import vpi, rocm, ext_dev
from .module import load_module, enabled, system_lib, load_static_library, num_threads
from .container import String, ShapeTuple
from .object_generic import const
from .params import (
    save_param_dict,
    load_param_dict,
    save_param_dict_to_file,
    load_param_dict_from_file,
)

from . import disco

from .support import _regex_match
from ..ffi import convert, dtype as DataType, DataTypeCode


def can_implicit_convert(src_dtype, dst_dtype):
    """Whether the "src_dtype" can be converted to "dst_dtype" implicitly."""
    src_dtype, dst_dtype = DataType(src_dtype), DataType(dst_dtype)

    if (
        src_dtype.lanes != dst_dtype.lanes
        or (src_dtype.is_vector and dst_dtype.bits < src_dtype.bits)
        or (dst_dtype.is_integer and src_dtype.is_float)
    ):
        return False

    return True


_DTYPE2RANGE = {"bool": (0, 1), "int8": (-128, 127), "uint8": (0, 255)}
_DTYPE2RANGE.update({"int16": (-32768, 32767), "uint16": (0, 65535)})
_DTYPE2RANGE.update({"int32": (-2147483648, 2147483647), "uint32": (0, 4294967295)})
_DTYPE2RANGE.update({"float16": (-65504.0, 65504.0)})
_DTYPE2RANGE.update({"float32": (-3.4028234663852886e38, 3.4028234663852886e38)})
_DTYPE2RANGE.update({"bfloat16": (-3.38953e38, 3.38953e38)})


def get_range(dtype):
    """Get the minimum and maximum value of the given data type."""
    dtype = DataType(dtype)
    elem_dtype = dtype.element_of

    if elem_dtype not in _DTYPE2RANGE and dtype.is_integer:
        bits = dtype.bits
        min_val = 0 if dtype.is_uint else -(2 ** (bits - 1))
        max_val = (2**bits - 1) if dtype.is_uint else (2 ** (bits - 1) - 1)
        _DTYPE2RANGE[elem_dtype] = (min_val, max_val)

    return _DTYPE2RANGE[elem_dtype]


def int_within_range(x, dtype):
    """Whether the given integer value "x" is in the range of the specified integer data type."""
    min_value, max_value = get_range(dtype)
    return min_value <= x <= max_value
