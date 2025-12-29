# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vmml_vmma_func(name, vdtype):
    sdot_func = getattr(S, name)

    @S.prim_func
    def vmml_vmma_func(
        a: S.ptr(vdtype, "global"),
        b: S.ptr(vdtype, "global"),
        c: S.ptr("fp32x8", "global"),
        d: S.ptr("fp32x8", "global"),
    ):
        cc = S.vconcat((c[0], c[1]))
        sdot_func(cc.addr, a[0], b[0])
        cc0, cc1 = S.vsplit(cc)
        d[0] = cc0
        d[1] = cc1

    return vmml_vmma_func


@pytest.mark.NOT_X3P
@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
@pytest.mark.parametrize("name", ("vmml", "vmma"))
def test_vmml_vmma(name, dtype):
    out_dtype = "float32"
    shape = (4, 4)
    a, b = rand(shape, dtype), rand(shape, dtype)
    c = rand(shape, out_dtype)
    gt_out = np.dot(a.astype(out_dtype), b.astype(out_dtype).transpose((1, 0)))
    if name == "vmma":
        gt_out += c

    vdtype = hw_native_vdtype(dtype)
    py_func = gen_vmml_vmma_func(name, vdtype)
    bm = BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(shape, dtype=out_dtype)
    py_func(a, b, c, py_out)
    if vdtype.is_float16:
        assert_allclose(py_out, gt_out, rtol=1e-3)

    npu_out = np.empty(shape, dtype=out_dtype)
    ex(a, b, c, npu_out)
    if vdtype.is_float16:
        assert_allclose(npu_out, gt_out, rtol=1e-3)

    assert_allclose(npu_out, py_out)


if __name__ == "__main__":
    test_vmml_vmma("vmml", "bfloat16")
    test_vmml_vmma("vmma", "bfloat16")
