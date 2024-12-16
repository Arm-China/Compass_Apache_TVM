# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import aipu
from tvm.aipu import script as S, testing


def gen_vmml_vmma_func(name):
    sdot_func = getattr(S, name)

    @S.prim_func
    def vmml_vmma_func(
        a: S.ptr("fp16x16", "global"),
        b: S.ptr("fp16x16", "global"),
        c: S.ptr("fp32x8", "global"),
        d: S.ptr("fp32x8", "global"),
    ):
        cc = S.vconcat(c[0], c[1])
        sdot_func(cc.addr, a[0], b[0])
        cc0, cc1 = S.vsplit(cc)
        d[0] = cc0
        d[1] = cc1

    return vmml_vmma_func


@pytest.mark.parametrize("name", ("vmml", "vmma"))
def test_vmml_vmma(name):
    dtype, out_dtype = "float16", "float32"
    shape = (4, 4)
    # TODO(aipu_tvm): Fix the random bug.
    # a, b = rand(shape, dtype), rand(shape, dtype)
    # c = rand(shape, out_dtype)
    a = np.array(range(16), dtype=dtype).reshape(shape)
    b = np.array(range(16), dtype=dtype).reshape(shape)
    c = np.array(range(16), dtype=out_dtype).reshape(shape)
    gt_out = np.dot(a.astype(out_dtype), b.astype(out_dtype).transpose((1, 0)))
    if name == "vmma":
        gt_out += c

    py_func = gen_vmml_vmma_func(name)
    bm = aipu.tir.BuildManager()
    ex = bm.build(py_func)

    py_out = np.empty(shape, dtype=out_dtype)
    py_func(a, b, c, py_out)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty(shape, dtype=out_dtype)
    ex(a, b, c, aipu_out)
    testing.assert_allclose(aipu_out, gt_out)


if __name__ == "__main__":
    test_vmml_vmma("vmml")
    test_vmml_vmma("vmma")
