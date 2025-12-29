# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import re
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def gen_vmul_to_vsl(vdtype, mask):
    lanes = vdtype.lanes

    @S.prim_func
    def vmul_to_vsl_func(a: S.ptr(vdtype, "global"), b: S.ptr(vdtype, "global"), c: S.ptr("fp32", "global")):
        b[0] = 4 * a[0]
        b[1] = S.vmul(a[1], 8, mask[lanes : lanes * 2])

        y = S.cast(2, vdtype)
        b[2] = S.vmul(a[2], y, mask[lanes * 2 :])

        c[:lanes] = S.fp32(S.i32(a[3])) * 8.2

    return vmul_to_vsl_func


def get_gt_out(dtype, lanes, a):
    ret = np.zeros(lanes * 3, dtype=dtype)
    ret[:lanes] = a[:lanes] * 4
    ret[lanes : lanes * 2] = a[lanes : lanes * 2] * 8
    ret[lanes * 2 :] = a[lanes * 2 : lanes * 3] * 2
    return ret, a[lanes * 3 :].astype("int32").astype("float32") * 8.2


@pytest.mark.parametrize("dtype", ("int16", "uint16", "int32", "uint32"))
def test_opt_vmul_to_vsl(dtype):
    vdtype = hw_native_vdtype(dtype)
    lanes = vdtype.lanes
    a = rand(lanes * 4, dtype)
    mask = rand(lanes * 3, "bool")
    mask[:lanes] = True
    # Add False to avoid simplify the expected vlsl statement.
    mask[lanes + 1] = False
    mask[lanes * 2 + 1] = False
    gt_out0, gt_out1 = get_gt_out(dtype, lanes, a)

    py_func = gen_vmul_to_vsl(vdtype, mask)
    bm = BuildManager()
    ex = bm.build(py_func)

    expects = (
        r"b\[0\] = __vlsl\(a\[0\], .*2\)",
        r"b\[1\] = __vlsl\(a\[1\], .*3\)",
        r"b\[2\] = __vlsl\(a\[2\], .*1\)",
        r".*a\[3\].* \* \(float8\).*8\.2.*\)",
    )
    for expect in expects:
        matches = re.search(expect, ex.c_code, re.MULTILINE)
        assert matches is not None, f"\nExpect snippet:\n{expect}\n\nCompass C code:\n{ex.c_code}\n"

    npu_out0, npu_out1 = np.empty(lanes * 3, dtype=dtype), np.empty(lanes, dtype="float32")
    ex(a, npu_out0, npu_out1)
    assert_allclose(npu_out0[mask], gt_out0[mask])
    assert_allclose(npu_out1, gt_out1)


if __name__ == "__main__":
    test_opt_vmul_to_vsl("uint16")
    test_opt_vmul_to_vsl("int32")
