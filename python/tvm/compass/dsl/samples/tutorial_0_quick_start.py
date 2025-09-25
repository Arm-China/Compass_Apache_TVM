# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, get_rpc_session
from tvm.compass.dsl.testing import rand, assert_allclose


dtype = "float32"


@S.prim_func
def func_add(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global"), n: S.i32):
    for i in range(n):
        c[i] = a[i] + b[i]


def test_vector_add():
    n = 1024
    a, b = rand(n, dtype), rand(n, dtype)
    gt_out = a + b

    print("1. Build...")
    bm = BuildManager()
    ex = bm.build(func_add)
    print("===============   PASS !  ===============\n")

    print("2. Run in Python (PySim)...")
    py_out = np.zeros(n, dtype)
    func_add(a, b, py_out, n)
    assert_allclose(py_out, gt_out)
    print("===============   PASS !  ===============\n")

    print("3. Run on NPU simulator...")
    npu_out = np.zeros(n, dtype)
    ex(a, b, npu_out, n)
    assert_allclose(npu_out, gt_out)
    print("===============   PASS !  ===============\n")

    print("Switch to execute on hardware device through RPC\n")
    # Change parameter "rpc_key" to use your expected hardware device,
    # "None" means get it from the environment variable "CPS_TVM_RPC_KEY".
    ex.rpc_sess = get_rpc_session(session_timeout=300, rpc_key=None)

    print("4. Run on remote hardware device...")
    npu_out = np.zeros(n, dtype)
    ex(a, b, npu_out, n)
    assert_allclose(npu_out, gt_out)
    print("===============   PASS !  ===============\n")

    print("5. Performence test through TVM...")
    print(ex.benchmark(a, b, npu_out, n))
    print("===============   PASS !  ===============\n")

    print("6. Performence test through NPU profiler...")
    ex.profile(a, b, npu_out, n)
    print("===============   PASS !  ===============\n")


if __name__ == "__main__":
    test_vector_add()
