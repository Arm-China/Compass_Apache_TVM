# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.analysis import has_perf_record_tick


@S.prim_func
def profiler_func(a: S.ptr("fp16", "global"), b: S.ptr("fp16", "global")):
    a_lsram_ptr = S.alloc(10, "fp16", scope="lsram")
    tid = S.get_local_id()
    if tid == 0:
        S.dma_copy(a_lsram_ptr, a, 10)
    S.barrier()

    S.perf_tick_begin(111)
    b[tid] = a_lsram_ptr[tid] + 1
    S.perf_tick_end(111)


def test_has_perf_record_tick():
    ir_mod = BuildManager().lower(profiler_func)
    assert has_perf_record_tick(ir_mod)


def test_has_no_perf_record_tick():
    @S.prim_func
    def func(a: S.ptr("fp16", "global"), b: S.ptr("fp16", "global")):
        b[0] = a[0]

    ir_mod = BuildManager().lower(func)
    assert not has_perf_record_tick(ir_mod)


if __name__ == "__main__":
    test_has_perf_record_tick()
    test_has_no_perf_record_tick()
