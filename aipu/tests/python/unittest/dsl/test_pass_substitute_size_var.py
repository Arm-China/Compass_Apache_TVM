# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
from tvm import aipu
from tvm.aipu import script as S


@S.prim_func
def primfunc():
    tec_num = S.get_local_size()
    tid = S.get_local_id()
    n = 127

    per_size = S.ceildiv(n, tec_num)
    input_offset = per_size * tid
    each_size = S.clip(n - input_offset, 0, per_size)

    if each_size % n > 0:
        return


def test_pass_substitute_size_var():
    bm = aipu.tir.BuildManager()
    ex = bm.build(primfunc)

    expect = "if (0 < (each_size % 127))"
    assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nAIPU C code:\n{ex.c_code}\n"


if __name__ == "__main__":
    test_pass_substitute_size_var()
