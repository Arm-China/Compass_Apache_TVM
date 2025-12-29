# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
from tvm.compass.dsl import BuildManager, script as S


@S.prim_func
def while_func(init: S.int32):
    i = init
    while i < 10:
        i = i + 2


def test_while_func():
    bm = BuildManager()
    ex = bm.build(while_func)
    print(ex.c_code)
    """
    __kernel void while_func0(int init) {
        int i = init;
        while ((i < 10)){
            i = (i + 2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    """
    expects = (
        "while ((i < 10))",
        "i = (i + 2);",
    )
    for expect in expects:
        assert expect in ex.c_code, f"\nExpect snippet:\n{expect}\n\nCompass C code:\n{ex.c_code}\n"


if __name__ == "__main__":
    test_while_func()
