# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import te
from tvm.compass.dsl import BuildManager
from tvm.compass.dsl.testing import assert_allclose


def test_attr_mcpu():
    dtype = "uint8"
    dshape = (32,)
    A = te.placeholder(dshape, name="A", dtype=dtype)
    B = te.placeholder(dshape, name="B", dtype=dtype)
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    tir_func = te.create_prim_func([A, B, C])

    a = np.ones(dshape, dtype)
    b = np.ones(dshape, dtype)

    bm = BuildManager(target="compass -mcpu=X2_1204")
    ex = bm.build(tir_func, name="fadd")

    c = np.empty(dshape, dtype)
    ex.run(a, b, c)
    assert_allclose(c, a + b)


@pytest.mark.X1
def test_mcpu_fail():
    with pytest.raises(AssertionError) as exc_info:
        BuildManager(target="compass -mcpu=X1_1204")

    exc_msg = str(exc_info.value)
    expect = 'The Compass DSL does not support the target "X1_1204"'
    assert expect in exc_msg, f"\nExpect snippet:\n{expect}\n\nException Message:\n{exc_msg}\n"


if __name__ == "__main__":
    test_attr_mcpu()
    test_mcpu_fail()
