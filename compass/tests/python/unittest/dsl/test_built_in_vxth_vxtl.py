# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose
from tvm.compass.dsl.utils import double_elem_width


def gen_vxt(sdot_func, vdtype, out_vdtype):
    @S.prim_func
    def vxt_func(x: S.ptr(vdtype, "global"), out: S.ptr(out_vdtype, "global")):
        out[0] = sdot_func(x[0])

    return vxt_func


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16"))
@pytest.mark.parametrize("func_name", ("vxtl", "vxth"))
def test_vxt(dtype, func_name):
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    x = rand(n, dtype)
    out_vdtype = double_elem_width(vdtype)
    out_dtype = out_vdtype.element_of
    out_n = out_vdtype.lanes

    gt_out = (x[:out_n] if func_name == "vxtl" else x[out_n:]).astype(out_dtype)

    f_vxt = gen_vxt(getattr(S, func_name), vdtype, out_vdtype)
    bm = BuildManager()
    ex = bm.build(f_vxt, name=func_name)

    py_out = np.empty(out_n, out_dtype)
    f_vxt(x, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(out_n, out_dtype)
    ex(x, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vxt(dtype="int8", func_name="vxtl")
