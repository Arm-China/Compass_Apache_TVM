# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, CompassInfo
from tvm.compass.dsl.testing import rand


@S.prim_func
def vcast_no_zip_func(inp: S.ptr("float16", "global"), out: S.ptr("float32", "global")):
    va = S.vload(inp)
    va32 = S.tag(S.cast(va, "float32"), "no_zip")
    S.vstore(S.vrpadd(va32), out, "1T")


def test_vcast_no_zip():
    dtype = "float16"
    out_dtype = "float32"
    n = 16
    a = rand(n, dtype)
    a32 = a.astype(out_dtype)
    sum_in_order = np.sum(a32[:8]) + np.sum(a32[8:])
    sum_in_even_odd = np.sum(a32[::2]) + np.sum(a32[1::2])
    cps_info = CompassInfo.get("X2_1204")
    gt_npu = sum_in_even_odd if cps_info.version == "X2" else sum_in_order

    bm = BuildManager()
    ex = bm.build(vcast_no_zip_func)

    py_out = np.empty(n, dtype=out_dtype)
    vcast_no_zip_func(a, py_out)
    assert py_out[0] == sum_in_order

    npu_out = np.empty(n, dtype=out_dtype)
    ex(a, npu_out)
    assert npu_out[0] == gt_npu

    unexpect_snippets = ["__vzipl", "__vziph"]
    c_code = ex.c_code.strip()
    for unexpect_snippet in unexpect_snippets:
        msg = f'Unexpect snippet "{unexpect_snippet}" in Compass C Code:\n{c_code}\n'
        assert unexpect_snippet not in c_code, msg


if __name__ == "__main__":
    test_vcast_no_zip()
