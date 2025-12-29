# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import assert_allclose


@S.prim_func
def asm_func(a: S.ptr("i8x32", "global"), b: S.ptr("i8x32", "global")):
    x = a[0]
    y = S.i8x32(0)
    S.asm(
        # fmt: off
        "{add t0.b, %[inp].b, 1;\n\t}\n\t"
        "{add %[out].b, %[inp].b, t0.b;}",
        # fmt: on
        outputs={"out": ("=&t", y)},
        inputs={"inp": ("t", x)},
        clobbers=["t0"],
        qualifiers="volatile",
    )
    b[0] = y


def test_inline_asm():
    dtype = "int8"
    n = 32
    a = np.array(range(n), dtype)
    gt_out = a + a + 1

    bm = BuildManager()
    ex = bm.build(asm_func)

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_inline_asm()
