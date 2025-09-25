# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import te
from tvm.compass.dsl import BuildManager, get_rpc_session
from tvm.compass.dsl.testing import assert_allclose, clear_traceback


def test_args_order():
    dtype = "uint8"
    dshape = (32,)
    A = te.placeholder(dshape, name="A", dtype=dtype)
    B = te.placeholder(dshape, name="B", dtype=dtype)
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    tir_func = te.create_prim_func([A, B, C])

    bm = BuildManager()

    ir_mod = bm.lower(tir_func, name="fadd")
    print(ir_mod)

    a = np.ones(dshape, dtype)
    b = np.ones(dshape, dtype)

    c = np.empty(dshape, dtype)
    ex = bm.build(tir_func, name="fadd")
    ex.run(a, b, c)
    assert_allclose(c, a + b)

    tir_func = te.create_prim_func([C, A, B])
    c = np.empty(dshape, dtype)
    ex = bm.build(tir_func, name="add")
    ex.run(c, a, b)
    assert_allclose(c, a + b)

    tir_func = te.create_prim_func([B, C, A])
    c = np.empty(dshape, dtype)
    ex = bm.build(tir_func, name="add")
    ex.run(b, c, a)
    assert_allclose(c, a + b)


@pytest.mark.NOT_X1
@pytest.mark.REQUIRE_RPC
@clear_traceback
def test_rpc_corner_case():
    dtype = "uint8"
    dshape = (32,)
    A = te.placeholder(dshape, name="A", dtype=dtype)
    B = te.placeholder(dshape, name="B", dtype=dtype)
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    tir_func = te.create_prim_func([A, B, C])

    bm = BuildManager()
    ex = bm.build(tir_func, name="add")

    ex.rpc_sess = get_rpc_session()
    ex.mtriple = "aarch64-linux-gnu"

    a = np.ones(dshape, dtype)
    b = np.ones(dshape, dtype)
    c = np.empty(dshape, dtype)
    ex.run(a, b, c)
    assert_allclose(c, a + b)


@pytest.mark.NOT_X1
@pytest.mark.REQUIRE_RPC
@clear_traceback
def test_executor():
    dtype = "uint8"
    dshape = (32,)
    A = te.placeholder(dshape, name="A", dtype=dtype)
    B = te.placeholder(dshape, name="B", dtype=dtype)
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    tir_func = te.create_prim_func([A, B, C])

    bm = BuildManager()
    ex = bm.build(tir_func, name="fadd")

    a = np.ones(dshape, dtype)
    b = np.ones(dshape, dtype)

    # sim: __call__
    c = np.empty(dshape, dtype)
    ex(a, b, c)
    assert_allclose(c, a + b)

    # sim: run
    c = np.empty(dshape, dtype)
    ex.run(a, b, c)
    assert_allclose(c, a + b)

    # sim: benchmark
    c = np.empty(dshape, dtype)
    print(ex.benchmark(a, b, c))
    assert_allclose(c, a + b)

    # Switch to execute on hardware device through RPC.
    ex.rpc_sess = get_rpc_session()

    # rpc: __call__
    c = np.empty(dshape, dtype)
    ex(a, b, c)
    assert_allclose(c, a + b)

    # rpc: run
    c = np.empty(dshape, dtype)
    ex.run(a, b, c)
    assert_allclose(c, a + b)

    # rpc: benchmark
    c = np.empty(dshape, dtype)
    print(ex.benchmark(a, b, c))
    assert_allclose(c, a + b)


if __name__ == "__main__":
    test_args_order()
    test_rpc_corner_case()
    test_executor()
