# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import numpy as np
from tvm.compass.dsl import BuildManager, script as S
from tvm.compass.dsl.testing import rand, assert_allclose


HALF_INT8_ELEMS_ON_LSRAM = 30 * 1024 // 2


def gen_eltwise_add(dtype):
    @S.prim_func
    def compute(
        inp0_lsram: S.ptr(dtype, "lsram"),
        inp1_lsram: S.ptr(dtype, "lsram"),
        size: S.i32,
        zp_i0: S.i16,
        zp_i1: S.i16,
        zp_o: S.i16,
        scale_i0: S.u16,
        scale_i1: S.u16,
        scale_o: S.u8,
        shift: S.i8,
    ):
        for i in range((size + 15) // 16):
            idx_base = i * 16
            # Load 16 int8 elements each time and cast to int32.
            a32 = S.cast(inp0_lsram[idx_base : idx_base + 16], "int32")
            b32 = S.cast(inp1_lsram[idx_base : idx_base + 16], "int32")

            # Compute: (a + zp_i0) * scale_i0.
            # ==================================================
            # Calculate the zero point and scale for first input. The zero point and scale of input0 are
            # derived from the quantization stage. Here calculation follows the requirements and order
            # of the quantization stage to ensure accuracy is not lost.
            # The addition here does not use saturation operation. This is because the value contained
            # in the 32-bit type "a32" is 8-bit, and the zero point "zp_i0" is 8-bit. The final result
            # will not exceed 9 bits.
            # Considering result of addition is signed value, thus, sign of output for multiplication is
            # "s".
            a32_add = a32 + zp_i0
            a32_mul = S.vmul(a32_add, scale_i0, out_sign="s")

            # Compute: (b + zp_i1) * scale_i1.
            # ==================================================
            # Calculate the zero point and scale for second input.
            b32_add = b32 + zp_i1
            b32_mul = S.vmul(b32_add, scale_i1, out_sign="s")

            # Element-wise with method=ADD.
            # ==================================================
            # The addition here does not use saturation operation. This is because the value contained
            # in the 32-bit variable "a32_mul" is at most 25 bits. The sum of two values, each not exceeding
            # 25 bits, will not exceed 26 bits.
            tmp_w_add = a32_mul + b32_mul

            # Multiply with uint8 output scale.
            tmp_w_mul = S.vmul(tmp_w_add, scale_o, out_sign="s")

            # Shift and cast from 32-bit to 16-bit.
            # ==================================================
            # Shift to right when shift > 0, otherwise to left.
            tmp_h = S.i16x16(0)
            if shift < 0:
                tmp_h = S.cast(S.vsl(tmp_w_mul, -shift), "int16", saturate=True)
            else:
                tmp_h = S.cast(S.vsr(tmp_w_mul, shift, with_round=True), "int16", saturate=True)

            # Subtract zero point of output.
            tmp_h = S.vsub(tmp_h, zp_o, saturate=True, out_sign="s")

            # Cast from 16-bit to 8-bit and save 16 int8 elements each time.
            inp0_lsram[idx_base : idx_base + 16] = S.cast(tmp_h, dtype, saturate=True)

    @S.prim_func(is_entry=True)
    def eltwise_add_func(
        a: S.ptr(dtype, "global"),
        b: S.ptr(dtype, "global"),
        c: S.ptr(dtype, "global"),
        n: S.i32,
        zp_i0: S.i16,
        zp_i1: S.i16,
        zp_o: S.i16,
        scale_i0: S.u16,
        scale_i1: S.u16,
        scale_o: S.u8,
        shift: S.i8,
    ):
        tec_num = S.get_local_size()
        tid = S.get_local_id()

        per_size = (n + tec_num - 1) // tec_num
        input_offset = tid * per_size
        each_size = S.clip(n - input_offset, min_val=0, max_val=per_size)

        if input_offset >= n:
            return

        a_lsram = S.alloc_buffer((HALF_INT8_ELEMS_ON_LSRAM,), dtype=dtype, scope="lsram")
        b_lsram = S.alloc_buffer((HALF_INT8_ELEMS_ON_LSRAM,), dtype=dtype, scope="lsram")

        for i in range(each_size // HALF_INT8_ELEMS_ON_LSRAM):
            base = input_offset + i * HALF_INT8_ELEMS_ON_LSRAM
            S.dma_copy(a_lsram, a + base, HALF_INT8_ELEMS_ON_LSRAM)
            S.dma_copy(b_lsram, b + base, HALF_INT8_ELEMS_ON_LSRAM)
            compute(a_lsram, b_lsram, HALF_INT8_ELEMS_ON_LSRAM, zp_i0, zp_i1, zp_o, scale_i0, scale_i1, scale_o, shift)
            S.dma_copy(c + base, a_lsram, HALF_INT8_ELEMS_ON_LSRAM)

        remain_size = each_size % HALF_INT8_ELEMS_ON_LSRAM
        if remain_size != 0:
            base = input_offset + each_size // HALF_INT8_ELEMS_ON_LSRAM * HALF_INT8_ELEMS_ON_LSRAM
            S.dma_copy(a_lsram, a + base, remain_size)
            S.dma_copy(b_lsram, b + base, remain_size)
            compute(a_lsram, b_lsram, remain_size, zp_i0, zp_i1, zp_o, scale_i0, scale_i1, scale_o, shift)
            S.dma_copy(c + base, a_lsram, remain_size)

    return eltwise_add_func


def get_gt(a, b, zp_i0, zp_i1, zp_o, scale_i0, scale_i1, scale_o, shift):
    a = a.astype("int32")
    b = b.astype("int32")

    a = (a + zp_i0) * scale_i0
    b = (b + zp_i1) * scale_i1
    out = a + b
    out = (a + b) * scale_o
    out = np.round(out * (0.5**shift)).astype("int32")
    out -= zp_o
    out = out.clip(np.iinfo("int8").min, np.iinfo("int8").max)
    return out.astype("int8")


def test_eltwise_add():
    # input data
    n = 3568
    dtype = "int8"
    scale_o, scale_i0, scale_i1 = (123, 1332, 2331)
    shift = 23
    zp_o, zp_i0, zp_i1 = (3, 5, 8)
    a = rand(n, dtype)
    b = rand(n, dtype)
    gt_out = get_gt(a, b, zp_i0, zp_i1, zp_o, scale_i0, scale_i1, scale_o, shift)

    # build the kernel
    py_func = gen_eltwise_add(dtype)
    bm = BuildManager()
    ex = bm.build(py_func)

    # run on PySim
    py_out = np.zeros((n,), dtype=dtype)
    py_func(a, b, py_out, n, zp_i0, zp_i1, zp_o, scale_i0, scale_i1, scale_o, shift)

    # run on Compass simulator
    npu_out = np.zeros((n,), dtype=dtype)
    ex(a, b, npu_out, n, zp_i0, zp_i1, zp_o, scale_i0, scale_i1, scale_o, shift)

    # verify result
    print(f"a[:4]       ={a[:4]}")
    print(f"b[:4]       ={b[:4]}")
    print(f"npu_out[:4]={npu_out[:4]}")
    print(f"gt_out[:4]  ={gt_out[:4]}")

    assert_allclose(py_out, gt_out)
    assert_allclose(npu_out, gt_out)
    print("=============== SUCCESS ! ===============")


if __name__ == "__main__":
    test_eltwise_add()
