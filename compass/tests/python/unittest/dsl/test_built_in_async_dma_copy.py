# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


size = 512
half_size = size // 2


def func_async_copy(dtype):
    @S.prim_func
    def func_dma_copy(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global")):
        a_lsram = S.alloc_buffer(dtype=dtype, shape=(size), scope="lsram")
        b_lsram = S.alloc_buffer(dtype=dtype, shape=(size), scope="lsram")
        c_lsram = S.alloc_buffer(dtype=dtype, shape=(size), scope="lsram")

        ev0, ev1, ev2 = S.alloc_events(3)
        S.async_dma_copy(a_lsram, a, half_size, event=ev0)
        S.async_dma_copy(b_lsram, b, half_size, event=ev1)
        S.wait_events(ev0, ev1)

        S.async_dma_copy(a_lsram.addr_of(half_size), a + half_size, half_size, event=ev0)
        S.async_dma_copy(b_lsram.addr_of(half_size), b + half_size, half_size, event=ev1)
        for i in range(half_size):
            c_lsram[i] = a_lsram[i] + b_lsram[i]
        S.async_dma_copy(c, c_lsram, half_size, event=ev2)

        S.wait_events(ev0, ev1)
        for i in range(half_size):
            c_lsram[i + half_size] = a_lsram[i + half_size] + b_lsram[i + half_size]
        S.async_dma_copy(c + half_size, c_lsram.addr_of(half_size), half_size, event=ev0)

        S.wait_events(ev0, ev2)
        S.free_events(ev0, ev1, ev2)

    return func_dma_copy


@pytest.mark.parametrize("dtype", ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16", "float32"))
def test_async_dma_copy(dtype):
    a = rand(size, dtype)
    b = rand(size, dtype)

    gt_out = a + b

    bm = BuildManager()
    prim_func = func_async_copy(dtype)
    ex = bm.build(prim_func)

    py_out = np.empty(size, dtype=dtype)
    prim_func(a, b, py_out)
    assert_allclose(py_out, gt_out)

    npu_out = np.empty(size, dtype=dtype)
    ex(a, b, npu_out)
    assert_allclose(npu_out, gt_out)


if __name__ == "__main__":
    test_async_dma_copy("int8")
