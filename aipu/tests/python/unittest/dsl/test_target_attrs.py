# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu, te, testing


def test_attr_mcpu():
    dtype = "uint8"
    dshape = (32,)
    A = te.placeholder(dshape, name="A", dtype=dtype)
    B = te.placeholder(dshape, name="B", dtype=dtype)
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    s = te.create_schedule(C.op)

    a = np.ones(dshape, dtype)
    b = np.ones(dshape, dtype)

    bm = aipu.tir.BuildManager(target="aipu -mcpu=X2_1204")
    ex = bm.build(s, [A, B, C], name="fadd")

    c = np.empty(dshape, dtype)
    ex.run(a, b, c)
    testing.assert_allclose(c, a + b)


@pytest.mark.X1_1204
def test_mcpu_fail():
    with pytest.raises(AssertionError) as exc_info:
        aipu.tir.BuildManager(target="aipu -mcpu=X1_1204")

    exc_msg = str(exc_info.value)
    expect = 'The Compass DSL does not support the target "X1_1204"'
    assert expect in exc_msg, f"\nExpect snippet:\n{expect}\n\nException Message:\n{exc_msg}\n"


if __name__ == "__main__":
    test_attr_mcpu()
    test_mcpu_fail()
