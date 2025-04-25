# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import tvm
from tvm import te, aipu


def get_schedule():
    dtype = "uint8"
    h = tvm.runtime.convert(128)
    w = tvm.runtime.convert(256)
    A = te.placeholder((h, w), name="A", dtype=dtype)
    B = te.placeholder((h, w), name="B", dtype=dtype)
    C = te.compute((h, w), lambda *i: A(*i) + B(*i), name="C")

    s = te.create_schedule(C.op)
    return (A, B, C, s)


def test_DMA1D_single_loop():
    A, B, C, s = get_schedule()
    al = s.cache_read(A, "lsram", C)
    bl = s.cache_read(B, "lsram", C)
    cl = s.cache_write(C, "lsram")

    c_axis = s[C].fuse(*C.op.axis)
    s[al].fuse(*al.op.axis)
    s[bl].fuse(*bl.op.axis)
    s[cl].fuse(*cl.op.axis)
    sram_axis, _ = s[C].split(c_axis, nparts=4)
    s[al].compute_at(s[C], sram_axis)
    s[bl].compute_at(s[C], sram_axis)
    s[cl].compute_at(s[C], sram_axis)
    s[C].bind(sram_axis, te.thread_axis("threadIdx.x"))

    bm = aipu.tir.BuildManager()
    ex = bm.build(s, [A, B, C], name="fadd")
    c_code = ex.c_code.strip()

    expect = """\
#include <aipu/tvm_aipu.h>

GEN_DMA_DIRECT_EXT2INT(kGlobal, kLsram);
GEN_DMA_DIRECT_INT2EXT(kLsram, kGlobal);

__kernel void fadd(__global uchar* restrict A, __global uchar* restrict B, __global uchar* restrict C);

__kernel void fadd(__global uchar* restrict A, __global uchar* restrict B, __global uchar* restrict C) {
  int tid = get_local_id(0);
  int cse_var_1 = (tid * 8192);
  __lsram uchar A_lsram[8192];
  __lsram uchar B_lsram[8192];
  DmaDirect_kGlobal_to_kLsram((int)A_lsram, (int)(A + cse_var_1), 8192, 8192, 8192, 8192);
  DmaDirect_kGlobal_to_kLsram((int)B_lsram, (int)(B + cse_var_1), 8192, 8192, 8192, 8192);
  for (int i0_c_i1_c_fused = 0; i0_c_i1_c_fused < 8192; i0_c_i1_c_fused += 1) {
    A_lsram[i0_c_i1_c_fused] = (uchar)(A_lsram[i0_c_i1_c_fused] + B_lsram[i0_c_i1_c_fused]);
  }
  DmaDirect_kLsram_to_kGlobal((int)(C + cse_var_1), (int)A_lsram, 8192, 8192, 8192, 8192);
  barrier(CLK_LOCAL_MEM_FENCE);
}
""".strip()
    assert expect == c_code, f"\nExpect snippet:\n{expect}\n\nAIPU C code:\n{c_code}\n"


def test_DMA1D_multiple_loop():
    A, B, C, s = get_schedule()
    al = s.cache_read(A, "lsram", C)
    bl = s.cache_read(B, "lsram", C)
    cl = s.cache_write(C, "lsram")

    sram_axis, _ = s[C].split(C.op.axis[0], nparts=4)
    s[al].compute_at(s[C], sram_axis)
    s[bl].compute_at(s[C], sram_axis)
    s[cl].compute_at(s[C], sram_axis)
    s[C].bind(sram_axis, te.thread_axis("threadIdx.x"))

    bm = aipu.tir.BuildManager()
    ex = bm.build(s, [A, B, C], name="fadd")
    c_code = ex.c_code.strip()

    expect = """\
#include <aipu/tvm_aipu.h>

GEN_DMA_DIRECT_EXT2INT(kGlobal, kLsram);
GEN_DMA_DIRECT_INT2EXT(kLsram, kGlobal);

__kernel void fadd(__global uchar* restrict A, __global uchar* restrict B, __global uchar* restrict C);

__kernel void fadd(__global uchar* restrict A, __global uchar* restrict B, __global uchar* restrict C) {
  int tid = get_local_id(0);
  int cse_var_1 = (tid * 8192);
  __lsram uchar A_lsram[8192];
  __lsram uchar B_lsram[8192];
  DmaDirect_kGlobal_to_kLsram((int)A_lsram, (int)(A + cse_var_1), 8192, 8192, 8192, 8192);
  DmaDirect_kGlobal_to_kLsram((int)B_lsram, (int)(B + cse_var_1), 8192, 8192, 8192, 8192);
  for (int i0_c = 0; i0_c < 32; i0_c += 1) {
    for (int i1_c = 0; i1_c < 256; i1_c += 1) {
      int cse_var_2 = ((i0_c * 256) + i1_c);
      A_lsram[cse_var_2] = (uchar)(A_lsram[cse_var_2] + B_lsram[cse_var_2]);
    }
  }
  DmaDirect_kLsram_to_kGlobal((int)(C + cse_var_1), (int)A_lsram, 8192, 8192, 8192, 8192);
  barrier(CLK_LOCAL_MEM_FENCE);
}
""".strip()
    assert expect == c_code, f"\nExpect snippet:\n{expect}\n\nAIPU C code:\n{c_code}\n"


def test_DMA2D_double_loop():
    A, B, C, s = get_schedule()

    al = s.cache_read(A, "lsram", C)
    bl = s.cache_read(B, "lsram", C)
    cl = s.cache_write(C, "lsram")

    in_o, _ = s[C].split(C.op.axis[-1], nparts=4)
    s[al].compute_at(s[C], in_o)
    s[bl].compute_at(s[C], in_o)
    s[cl].compute_at(s[C], in_o)
    s[C].reorder(in_o, C.op.axis[0])
    s[C].bind(in_o, te.thread_axis("threadIdx.x"))

    bm = aipu.tir.BuildManager()
    tir = bm.lower(s, [A, B, C], name="fadd")
    print(tir)


def test_DMA2D_multiple_loop():
    A, B, C, s = get_schedule()

    al = s.cache_read(A, "lsram", C)
    bl = s.cache_read(B, "lsram", C)
    cl = s.cache_write(C, "lsram")

    in_o, _ = s[C].split(C.op.axis[-1], nparts=4)
    out_o, out_i = s[C].split(C.op.axis[0], nparts=4)
    s[C].reorder(in_o, out_o, out_i)

    s[al].split(al.op.axis[0], nparts=4)
    s[bl].split(bl.op.axis[0], nparts=4)
    s[cl].split(cl.op.axis[0], nparts=4)
    s[al].compute_at(s[C], in_o)
    s[bl].compute_at(s[C], in_o)
    s[cl].compute_at(s[C], in_o)
    s[C].bind(in_o, te.thread_axis("threadIdx.x"))

    bm = aipu.tir.BuildManager()
    tir = bm.build(s, [A, B, C], name="fadd")
    print(tir.c_code)


if __name__ == "__main__":
    test_DMA1D_single_loop()
    test_DMA1D_multiple_loop()
    # test_DMA2D_double_loop()
    # test_DMA2D_multiple_loop()
