# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
from tvm.compass.dsl import BuildManager, script as S, transform as compass_transform


@S.prim_func
def func_imm_var(buf: S.ptr("float32", "global")):
    data = 100.0
    var_23 = 0
    var_25 = 8
    var_26 = 256
    var_22 = 1
    for i in range(var_23, var_23 + (var_25 - var_23), var_22):
        for j in range(var_23, var_23 + (var_26 - var_23), var_22):
            buf[i * var_26 + j] = data


def test_pass_fold_imm_const():
    bm = BuildManager()
    mod = bm._parse(func_imm_var, "func_imm_var")
    mod = compass_transform.FoldConstant()(mod)

    expects = (
        'for i in T.serial(T.Sub(8, 0), annotations={"step": 1})',
        'for j in T.serial(T.Sub(256, 0), annotations={"step": 1}):',
        "buf_buf[i * 256 + j] = T.float32(100.0)",
    )
    for expect in expects:
        assert expect in str(mod), f"\nExpect snippet:\n{expect}\n\nmodule:\n{str(mod)}\n"


if __name__ == "__main__":
    test_pass_fold_imm_const()
