# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_reinterpret_func(scalar_imm, src_vdtype, dst_vdtype):
    src_dtype, dst_dtype = src_vdtype.element_of, dst_vdtype.element_of

    @S.prim_func
    def reinterpret_func(a: S.ptr(src_vdtype, "global"), out: S.ptr(dst_vdtype, "global")):
        # vector
        out[0] = S.reinterpret(a[0], dst_vdtype)

        # scalar
        cur_out, cur_a = (out + 1).as_ptr(dst_dtype), (a + 1).as_ptr(src_dtype)
        for i in range(src_vdtype.lanes):
            cur_out[i] = S.reinterpret(cur_a[i], dst_dtype)

        if src_vdtype.bits == 32:
            cur_out[src_vdtype.lanes] = S.reinterpret(scalar_imm, dst_dtype)

    return reinterpret_func


DTYPE_TUPLE = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float32", "float16", "bfloat16")


@pytest.mark.parametrize("src_dtype", DTYPE_TUPLE)
@pytest.mark.parametrize("dst_dtype", DTYPE_TUPLE)
def test_reinterpret(src_dtype, dst_dtype):
    src_vdtype, dst_vdtype = hw_native_vdtype(src_dtype), hw_native_vdtype(dst_dtype)
    if src_vdtype.bits != dst_vdtype.bits:
        pytest.skip("Reinterpret only supports bit-equal dtypes.")

    name = f"reinterpret_{src_dtype}_to_{dst_dtype}"
    print(f"==> {name}")
    n = src_vdtype.lanes * 2
    n = n + 1 if src_vdtype.bits == 32 else n
    a = rand(n, src_dtype)
    gt_out = a.view(dst_dtype)

    prim_func = gen_reinterpret_func(a[-1].tolist(), src_vdtype, dst_vdtype)
    bm = BuildManager()
    ex = bm.build(prim_func, name=name)

    py_out = np.empty(n, dst_dtype)
    prim_func(a, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(n, dst_dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_reinterpret(src_dtype="uint8", dst_dtype="int8")
    test_reinterpret(src_dtype="float32", dst_dtype="float32")
    test_reinterpret(src_dtype="float32", dst_dtype="int32")
