# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import get_range
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def get_vnsr_gt_out(x, shift, out_dtype):
    # with_round
    out = np.round(x * (0.5 ** shift.astype("uint32"))).astype("int64")
    # saturate
    out = np.clip(out, *get_range(out_dtype))
    return out.astype(out_dtype)


def gen_vnsrsr_func(dtype, out_dtype, hw_lanes, to_h):
    lanes0 = hw_lanes
    lanes1 = 2 * hw_lanes
    lanes2 = 3 * hw_lanes
    lanes3 = 4 * hw_lanes
    if hw_lanes == 8 and not to_h:
        f_concat = lambda x: S.vconcat((S.vconcat((x, x), "even"), S.vconcat((x, x), "even")), "even")
    else:
        f_concat = lambda x: S.vconcat((x, x), "even")

    @S.prim_func
    def vnsrsr_func(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), out: S.ptr(out_dtype, "global")):
        cur_a, cur_b, cur_out = a, b, out
        # 1. n
        va0 = S.vload(cur_a, lanes=lanes0)
        vb0 = S.vload(cur_b, lanes=lanes0)
        vnsr0 = S.vnsrsr(va0, vb0, to_h=to_h)
        vout0 = f_concat(vnsr0)
        S.vstore(vout0, cur_out, f"{lanes0}T")

        cur_a, cur_b, cur_out = cur_a + lanes0, cur_b + lanes0, cur_out + lanes0
        # 2. 2 * n
        va1 = S.vload(cur_a, lanes=lanes1)
        vb1 = S.vload(cur_b, lanes=lanes1)
        vnsr1 = S.vnsrsr(va1, vb1, to_h=to_h)
        vout1 = f_concat(vnsr1)
        S.vstore(vout1, cur_out, f"{lanes1}T")

        cur_a, cur_b, cur_out = cur_a + lanes1, cur_b + lanes1, cur_out + lanes1
        # 3. 3 * n
        va2 = S.vload(cur_a, lanes=lanes2)
        vb2 = S.vload(cur_b, lanes=lanes2)
        vnsr2 = S.vnsrsr(va2, vb2, to_h=to_h)
        vout2 = f_concat(vnsr2)
        S.vstore(vout2, cur_out, f"{lanes2}T")

        cur_a, cur_b, cur_out = cur_a + lanes2, cur_b + lanes2, cur_out + lanes2
        # 4. 4 * n
        va3 = S.vload(cur_a, lanes=lanes3)
        vb3 = S.vload(cur_b, lanes=lanes3)
        vnsr3 = S.vnsrsr(va3, vb3, to_h=to_h)
        vout3 = f_concat(vnsr3)
        S.vstore(vout3, cur_out, f"{lanes3}T")

    return vnsrsr_func


@pytest.mark.parametrize("dtype", ("int16", "uint16", "int32", "uint32"))
def test_vnsrsr(dtype):
    vdtype = hw_native_vdtype(dtype)
    hw_lanes = vdtype.lanes
    n = 10 * hw_lanes
    a = rand(n, dtype)
    shift = rand(n, dtype, low=0, high=vdtype.bits - 2)
    bm = BuildManager()
    for to_h in (True, False) if vdtype.bits == 32 else (False,):
        out_dtype = vdtype.with_bits(16 if to_h else 8).element_of
        gt_out = get_vnsr_gt_out(a, shift, out_dtype)

        py_func = gen_vnsrsr_func(dtype, out_dtype, hw_lanes, to_h)
        ex = bm.build(py_func)

        py_out = np.empty(n, dtype=out_dtype)
        py_func(a, shift, py_out)
        assert_allclose(py_out, gt_out)

        npu_out = np.empty(n, dtype=out_dtype)
        ex(a, shift, npu_out)
        assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_vnsrsr("uint32")
