# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
from tvm import tir, DataType
from tvm.compass.dsl import hw_native_vdtype
from tvm.compass.dsl.testing import rand
from tvm.compass.dsl.script.ir.utils import canonicalize_r, PARAM_R_MARK


def test_r_is_none():
    r = canonicalize_r(None, None)
    assert r is PARAM_R_MARK


@pytest.mark.parametrize("ret_dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_r_is_const_scalar(ret_dtype):
    ret_vdtype = hw_native_vdtype(ret_dtype)
    ret_vdtype_str = str(ret_vdtype)
    r_dtypes = {
        "int8": ("int8", "uint8"),
        "uint8": ("int8", "uint8"),
        "int16": ("int8", "uint8", "int16", "uint16"),
        "uint16": ("int8", "uint8", "int16", "uint16"),
        "int32": ("int8", "uint8", "int16", "uint16", "int32", "uint32"),
        "uint32": ("int8", "uint8", "int16", "uint16", "int32", "uint32"),
        "float16": ("float16",),
        "float32": ("float16", "float32"),
    }[ret_dtype]

    for r_dtype in r_dtypes:
        in_r = rand(1, r_dtype, return_python_type=True)
        out_r = canonicalize_r(in_r, ret_vdtype)
        assert out_r.dtype == ret_vdtype_str
        assert isinstance(out_r, tir.Call) and out_r.args[0] == "__vbcast"


@pytest.mark.parametrize("ret_dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_r_is_var(ret_dtype):
    ret_vdtype = hw_native_vdtype(ret_dtype)
    ret_vdtype_str = str(ret_vdtype)
    r_dtypes = {
        "int8": ("int8", "uint8"),
        "uint8": ("int8", "uint8"),
        "int16": ("int8", "uint8", "int16", "uint16"),
        "uint16": ("int8", "uint8", "int16", "uint16"),
        "int32": ("int8", "uint8", "int16", "uint16", "int32", "uint32"),
        "uint32": ("int8", "uint8", "int16", "uint16", "int32", "uint32"),
        "float16": ("float16",),
        "float32": ("float16", "float32"),
    }[ret_dtype]

    # 1. Test scalar var.
    for r_dtype in r_dtypes:
        out_r = canonicalize_r(tir.Var("scalar_r", r_dtype), ret_vdtype)
        assert out_r.dtype == ret_vdtype_str

    # 2. Test vector var.
    vector_r_dtypes = tuple(x for x in r_dtypes if DataType(x).bits == ret_vdtype.bits)
    for r_dtype in vector_r_dtypes:
        out_r = canonicalize_r(tir.Var("vector_r", hw_native_vdtype(r_dtype)), ret_vdtype)
        assert out_r.dtype == ret_vdtype_str


if __name__ == "__main__":
    test_r_is_none()
    test_r_is_const_scalar(ret_dtype="int8")
    test_r_is_const_scalar(ret_dtype="float16")
    test_r_is_var(ret_dtype="uint32")
    test_r_is_var(ret_dtype="float32")
