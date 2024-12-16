# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S
from tvm.aipu.utils import hw_native_vdtype, rand


def gen_debug_func(vdtype):

    # pylint: disable=multiple-statements, forgotten-debug-statement
    @S.prim_func
    def debug_func(a: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        breakpoint()
        out[0] += S.vadd(a[0], 1)

        # fmt: off
        import pdb; pdb.set_trace()  # noqa
        # fmt: on
        out[0] += S.vadd(a[0], 1)

        from pdb import set_trace

        set_trace()
        out[0] += S.vadd(a[0], 1)

    return debug_func


def test_debug():
    is_run_prim_func = False
    dtype = "int32"
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes

    prim_func = gen_debug_func(vdtype)
    ex = aipu.tir.BuildManager().build(prim_func)

    if is_run_prim_func:
        a = rand(n, dtype)

        py_out = np.empty(n, dtype=dtype)
        prim_func(a, py_out)

        aipu_out = np.empty(n, dtype=dtype)
        ex(a, aipu_out)


@S.prim_func
def print_func(a: S.ptr("i32", "global"), out: S.ptr("i32", "global")):
    out[0] = a[0]
    print(a[0])


def test_print_fail(capfd):
    with pytest.raises(RuntimeError):
        aipu.tir.BuildManager().build(print_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = 'The built-in "print" isn\'t supported, please use "S.printf".'
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


if __name__ == "__main__":
    test_debug()
    test_print_fail(None)
