# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_signed_or_unsigned_func(src_vdtype, dst_vdtype):
    sdot_func = S.i if dst_vdtype.is_int else S.u
    src_dtype, dst_dtype = src_vdtype.element_of, dst_vdtype.element_of

    @S.prim_func
    def s_u_func(a: S.ptr(src_vdtype, "global"), out: S.ptr(dst_vdtype, "global")):
        # vector
        out[0] = sdot_func(a[0])

        # scalar
        cur_out, cur_a = (out + 1).as_ptr(dst_dtype), (a + 1).as_ptr(src_dtype)
        for i in range(src_vdtype.lanes):
            cur_out[i] = sdot_func(cur_a[i])

    return s_u_func


@pytest.mark.parametrize("dtype", ("int8", "int16", "int32", "uint8", "uint16", "uint32"))
@pytest.mark.parametrize("to_signed", (True, False))
def test_signed_or_unsigned(dtype, to_signed):
    src_vdtype = hw_native_vdtype(dtype)
    dst_vdtype = src_vdtype.with_int() if to_signed else src_vdtype.with_uint()
    dst_dtype = dst_vdtype.element_of

    n = src_vdtype.lanes * 2
    a = rand(n, dtype)
    gt_out = a.astype(dst_dtype)

    py_func = gen_signed_or_unsigned_func(src_vdtype, dst_vdtype)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(n, dtype=dst_dtype)
    py_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dtype=dst_dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_signed_or_unsigned("int8", to_signed=False)
    test_signed_or_unsigned("uint32", to_signed=True)
