# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
from tvm.compass.dsl import BuildManager, script as S


@S.prim_func
def load(a: S.ptr("int32", "global")):
    buffer = S.alloc_buffer([1], dtype="int32")
    a[0] = buffer[0]
    scalar_var = S.int32(0)
    if a[0] > 1:
        scalar_var = buffer[0]
        a[0] = scalar_var


def test_load():
    bm = BuildManager()
    ex = bm.build(load)

    expects = (
        "a[0] = buffer[0];",
        "int scalar_var = 0;",
        "a[0] = scalar_var;",
    )
    for expect in expects:
        assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nCompass C code:\n{ex.c_code}\n"


@S.prim_func
def store(a: S.ptr("int32", "global")):
    buffer = S.alloc_buffer([1], dtype="int32")
    buffer[0] = a[0]
    scalar_var = S.int32(0)
    if a[0] > 1:
        scalar_var = buffer[0]
    a[0] = scalar_var


def test_store():
    bm = BuildManager()
    ex = bm.build(store)

    expects = (
        "buffer[0] = a[0];",
        "int scalar_var = 0;",
        "scalar_var = buffer[0];",
    )
    for expect in expects:
        assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nCompass C code:\n{ex.c_code}\n"


@S.prim_func
def vec_load(a: S.ptr("int32x8", "global"), b: S.i32):
    buffer = S.alloc_buffer([1], dtype="int32x8")
    a[0] = buffer[0]
    vector_var = S.int32x8(0)
    if b > 1:
        vector_var = buffer[0]
        a[0] = vector_var


def test_vec_load():
    bm = BuildManager()
    ex = bm.build(vec_load)

    expects = (
        "a[0] = buffer[0];",
        "int8 vector_var = (int8)0;",
        "a[0] = vector_var;",
    )
    for expect in expects:
        assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nCompass C code:\n{ex.c_code}\n"


@S.prim_func
def vec_store(a: S.ptr("int32x8", "global"), b: S.i32):
    buffer = S.alloc_buffer([1], dtype="int32x8")
    buffer[0] = a[0]
    vector_var = S.int32x8(0)
    if b > 1:
        vector_var = buffer[0]
    a[0] = vector_var


def test_vec_store():
    bm = BuildManager()
    ex = bm.build(vec_store)

    expects = (
        "buffer[0] = a[0];",
        "int8 vector_var = (int8)0;",
        "vector_var = buffer[0];",
    )
    for expect in expects:
        assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nCompass C code:\n{ex.c_code}\n"


if __name__ == "__main__":
    test_load()
    test_store()
    test_vec_load()
    test_vec_store()
