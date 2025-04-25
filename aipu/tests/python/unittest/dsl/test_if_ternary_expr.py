# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing


@S.prim_func
def py_func(a: S.ptr("int32", "global"), c: S.ptr("int32", "global")):
    a0, a1 = a[0], a[1]

    # general
    c[0] = a1 if a0 == 0 else 20
    c[1] = 20 if 0 >= a0 else 30
    c[2] = 20 if a0 != a1 else 30

    # complicated
    c[3] = a[3] if 10 - a1 > a[0] else a[4] + a[2]
    c[4] = (a[3] if 10 - a1 > a[0] else a[4]) + a[2]

    c[5] = (a[3] if 30 + 3 > 11 else a[4]) + a[5]
    c[6] = a[3] if 30 + 3 <= 11 else a[4] + a[5]

    vc_ptr = (c + 7).as_ptr("i32x8")
    va_ptr = a.as_ptr("i32x8")
    vc_ptr[0] = va_ptr[0] if a0 == 0 else 10


def get_gt_out(x, dtype):
    ret = np.empty(x.size, dtype)
    ret[0] = x[1] if x[0] == 0 else 20
    ret[1] = 20 if 0 >= x[0] else 30
    ret[2] = 20 if x[0] != x[1] else 30

    # complicated
    ret[3] = x[3] if 10 - x[1] > x[0] else x[4] + x[2]
    ret[4] = (x[3] if 10 - x[1] > x[0] else x[4]) + x[2]

    ret[5] = (x[3] if 30 + 3 > 11 else x[4]) + x[5]
    ret[6] = x[3] if 30 + 3 <= 11 else x[4] + x[5]

    ret[7:15] = x[:8] if x[0] == 0 else 10
    return ret


def test_if_ternary_expr():
    dtype = "int32"
    n = 15
    a = np.array(range(n), dtype)
    gt_out = get_gt_out(a, dtype)

    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    expects = (
        "bool cse_var_1 = (a0 == 0);",
        "c[0] = (cse_var_1 ? a1 : 20);",
        "c[1] = ((a0 <= 0) ? 20 : 30);",
        "c[2] = ((a0 != a1) ? 20 : 30);",
        "c[3] = (((a[0] + a1) < 10) ? a[3] : (a[4] + a[2]));",
        "c[4] = ((((a[0] + a1) < 10) ? a[3] : a[4]) + a[2]);",
        "c[5] = (a[3] + a[5]);",
        "c[6] = (a[4] + a[5]);",
        "vc_ptr[0] = (cse_var_1 ? va_ptr[0] : (int8)10);",
    )
    for expect in expects:
        assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nAIPU C code:\n{ex.c_code}\n"

    py_out = np.empty(n, dtype=dtype)
    py_func(a, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(n, dtype=dtype)
    ex(a, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


def test_compiling_compute():
    static_var = 8

    @S.prim_func
    def func(placeholder: S.ptr("int32", "global"), c: S.ptr("int32", "global")):
        c[0] = 0 if static_var == 8 else 16

    bm = aipu.tir.BuildManager()
    ex = bm.build(func)
    expect = "c[0] = 0;"
    assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nAIPU C code:\n{ex.c_code}\n"


if __name__ == "__main__":
    test_if_ternary_expr()
    test_compiling_compute()
