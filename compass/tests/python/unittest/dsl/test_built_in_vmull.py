# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand, assert_allclose


def vmull_int8_int16(n, vdtype0, vdtype1, out_dtype):
    out_sign = "u" if out_dtype.startswith("u") else "s"

    @S.prim_func
    def vmull_func(a: S.ptr(vdtype0, "global"), b: S.ptr(vdtype1, "global"), c: S.ptr(out_dtype, "global")):
        c[0 : n // 2] = S.vmull(a[0], b[0], out_sign=out_sign)

    return vmull_func


def vmull_int32(n, vdtype0, vdtype1, out_dtype):
    out_sign = "u" if out_dtype.startswith("u") else "s"

    @S.prim_func
    def vmull_func(a: S.ptr(vdtype0, "global"), b: S.ptr(vdtype1, "global"), c: S.ptr(out_dtype, "global")):
        c[0:n] = S.vmull(a[0], b[0], out_sign=out_sign)

    return vmull_func


def check_int32_vmull(gt, pred, n):
    for i in range(n // 2):
        assert gt[i] == pred[i * 2], f"Check int32 vmull failed, gt: {gt[i]}, pred: {pred[i * 2]}"


@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    (
        # ss-s
        ("int8", "int8", "int16"),
        ("int16", "int16", "int32"),
        ("int32", "int32", "int32"),
        # uu-u
        ("uint8", "uint8", "uint16"),
        ("uint16", "uint16", "uint32"),
        ("uint32", "uint32", "uint32"),
        # us-s
        ("uint8", "int8", "int16"),
        ("uint16", "int16", "int32"),
        ("uint32", "int32", "int32"),
        # us-u
        ("uint8", "int8", "uint16"),
        ("uint16", "int16", "uint32"),
        ("uint32", "int32", "uint32"),
        # su-s
        ("int8", "uint8", "int16"),
        ("int16", "uint16", "int32"),
        ("int32", "uint32", "int32"),
        # su-u
        ("int8", "uint8", "uint16"),
        ("int16", "uint16", "uint32"),
        ("int32", "uint32", "uint32"),
        # ss-u
        ("int8", "int8", "uint16"),
        ("int16", "int16", "uint32"),
        ("int32", "int32", "uint32"),
        # uu-s
        ("uint8", "uint8", "int16"),
        ("uint16", "uint16", "int32"),
        ("uint32", "uint32", "int32"),
    ),
)
def test_vmull(in0_dtype, in1_dtype, out_dtype):
    vdtype0 = hw_native_vdtype(in0_dtype)
    vdtype1 = hw_native_vdtype(in1_dtype)
    n = vdtype0.lanes
    a = rand(n, in0_dtype, low=0, high=127)
    b = rand(n, in1_dtype, low=0, high=127)
    gt_out = a.astype(out_dtype) * b.astype(out_dtype)
    # 8bit 16bit
    if vdtype0.bits != 32:
        prim_func = vmull_int8_int16(n, vdtype0, vdtype1, out_dtype)
        bm = BuildManager()
        ex = bm.build(prim_func, name="func_vmull_int8_16")

        py_out = np.empty(n, dtype=out_dtype)
        prim_func(a, b, py_out)
        assert_allclose(py_out[: n // 2], gt_out[: n // 2])

        npu_out = np.empty(n, dtype=out_dtype)
        ex(a, b, npu_out)
        assert_allclose(npu_out[: n // 2], gt_out[: n // 2])
    # 32bit
    else:
        prim_func = vmull_int32(n, vdtype0, vdtype1, out_dtype)
        bm = BuildManager()
        ex = bm.build(prim_func, name="func_vmull_int32")

        py_out = np.empty(n, dtype=out_dtype)
        prim_func(a, b, py_out)
        check_int32_vmull(gt_out, py_out, n)

        npu_out = np.empty(n, dtype=out_dtype)
        ex(a, b, npu_out)
        check_int32_vmull(gt_out, npu_out, n)


if __name__ == "__main__":
    test_vmull("uint8", "uint8", "int16")
