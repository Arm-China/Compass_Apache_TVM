# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
import pytest
import numpy as np
from tvm import get_range
from tvm.compass.dsl import BuildManager, script as S, hw_native_vdtype
from tvm.compass.dsl.testing import rand


def gen_printf(vdtype, printf_type):
    i16_min, i16_max = get_range("int16")
    _, u16_max = get_range("uint16")
    i32_min, i32_max = get_range("int32")
    _, u32_max = get_range("uint32")
    fp16_min, fp16_max = get_range("float16")
    fp32_min, fp32_max = get_range("float32")

    @S.prim_func
    def printf_d_i_u_f_func(a: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vadd(a[0], 0)
        tec_cnt = S.get_local_size()

        S.printf("Test %%d: tec_cnt=%d, signed expr:%d\n", tec_cnt, tec_cnt - 10)
        S.printf("Test %%d: %d, %d, %d\n", i32_min, i16_min, 0)
        S.printf("Test %%d: %d, %d, %d\n", i16_max, i32_max, u32_max)

        S.printf("Test %%i: tec_cnt=%i, signed expr:%i\n", tec_cnt, tec_cnt - 10)
        S.printf("Test %%i: %i, %i, %i\n", i32_min, i16_min, 0)
        S.printf("Test %%i: %i, %i, %i\n", i16_max, i32_max, u32_max)

        S.printf("Test %%u: tec_cnt=%u, signed expr:%u\n", tec_cnt, tec_cnt - 10)
        S.printf("Test %%u: %u, %u, %u\n", i32_min, i16_min, 0)
        S.printf("Test %%u: %u, %u, %u\n", i16_max, i32_max, u32_max)

        S.printf("Test %%f: %f, %f, %f, %f, %f\n", fp16_min, -1.234567, 0.0, 1.234567, fp16_max)
        S.printf("Test %%f: %f\n", 0.0000546)
        # Zhouyi NPU result wrong, got "-42949672954294967295.000000", "42949672954294967295.000000".
        S.printf("Test %%f: %f, %f\n", fp32_min, fp32_max)

    @S.prim_func
    def printf_o_x_X_func(a: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vadd(a[0], 0)
        tec_cnt = S.get_local_size()
        var_fp32 = S.fp32(1.25)

        S.printf("Test %%o: tec_cnt=%o, signed expr:%o\n", tec_cnt, tec_cnt - 10)
        S.printf("Test %%o: %o, %o, %o\n", i32_min, i16_min, 0)
        S.printf("Test %%o: %o, %o, %o\n", i16_max, i32_max, u32_max)

        S.printf("Test %%x: tec_cnt=%x, signed expr:%x\n", tec_cnt, tec_cnt - 10)
        S.printf("Test %%x: %x, %x, %x\n", i32_min, i16_min, 0)
        S.printf("Test %%x: %x, %x, %x\n", i16_max, i32_max, u32_max)
        S.printf("Test %%x: %x, %x\n", 1.25, var_fp32)

        S.printf("Test %%X: tec_cnt=%X, signed expr:%X\n", tec_cnt, tec_cnt - 10)
        S.printf("Test %%X: %X, %X, %X\n", i32_min, i16_min, 0)
        S.printf("Test %%X: %X, %X, %X\n", i16_max, i32_max, u32_max)
        S.printf("Test %%X: %X, %X\n", 1.25, var_fp32)

    @S.prim_func
    def printf_p_c_s_func(a: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vadd(a[0], 0)
        tec_cnt = S.get_local_size()

        # The "%p" is meaningless for PySim, will just leave it there.
        S.printf("Test %%p: address of the pointer a is %p\n", a)
        S.printf("Test %%c: %c\n", 79 + tec_cnt)
        # Current can't support string variable.
        S.printf("Test %%s: %s\n", "This is a literal string.")

    @S.prim_func
    def printf_vector_func(a: S.ptr(vdtype, "global"), out: S.ptr(vdtype, "global")):
        out[0] = S.vadd(a[0], 0)

        S.printf("Test %%v32hhx: %v32hhx\n", S.i8x32(-128))
        S.printf("Test %%v32hhx: %v32hhx\n", S.i8x32(0))
        S.printf("Test %%v32hhx: %v32hhx\n", S.i8x32(127))
        S.printf("Test %%v32hhx: %v32hhx\n", S.u8x32(255))

        S.printf("Test %%v16hx: %v16hx\n", S.i16x16(i16_min))
        S.printf("Test %%v16hx: %v16hx\n", S.i16x16(0))
        S.printf("Test %%v16hx: %v16hx\n", S.i16x16(i16_max))
        S.printf("Test %%v16hx: %v16hx\n", S.u16x16(u16_max))

        S.printf("Test %%v8hlx: %v8hlx\n", S.i32x8(i32_min))
        S.printf("Test %%v8hlx: %v8hlx\n", S.i32x8(0))
        S.printf("Test %%v8hlx: %v8hlx\n", S.i32x8(i32_max))
        S.printf("Test %%v8hlx: %v8hlx\n", S.u32x8(u32_max))
        S.printf("Test %%v8hlx: %v8hlx\n", S.fp32x8(1.25))

        S.printf("Test %%v16hf: %v16hf\n", S.fp16x16(fp16_min))
        S.printf("Test %%v16hf: %v16hf\n", S.fp16x16(0.0))
        S.printf("Test %%v16hf: %v16hf\n", S.fp16x16(1.2345678))
        S.printf("Test %%v16hf: %v16hf\n", S.fp16x16(fp16_max))

        S.printf("Test %%v8hlf: %v8hlf\n", S.fp32x8(fp32_min))
        S.printf("Test %%v8hlf: %v8hlf\n", S.fp32x8(0.0))
        S.printf("Test %%v8hlf: %v8hlf\n", S.fp32x8(1.2345678))
        S.printf("Test %%v8hlf: %v8hlf\n", S.fp32x8(fp32_max))

    if printf_type == "d_i_u_f":
        return printf_d_i_u_f_func
    if printf_type == "o_x_X":
        return printf_o_x_X_func
    if printf_type == "p_c_s":
        return printf_p_c_s_func

    assert printf_type == "vector"
    return printf_vector_func


@pytest.mark.parametrize("printf_type", ("d_i_u_f", "o_x_X", "p_c_s", "vector"))
def test_printf(printf_type, capfd):
    dtype = "int8"
    vdtype = hw_native_vdtype(dtype)
    n = vdtype.lanes
    a = rand(n, dtype)
    expects = {
        "d_i_u_f": (
            "Test %d: tec_cnt=4, signed expr:-6",
            "Test %d: -2147483648, -32768, 0",
            "Test %d: 32767, 2147483647, -1",
            "Test %i: tec_cnt=4, signed expr:-6",
            "Test %i: -2147483648, -32768, 0",
            "Test %i: 32767, 2147483647, -1",
            "Test %u: tec_cnt=4, signed expr:4294967290",
            "Test %u: 2147483648, 4294934528, 0",
            "Test %u: 32767, 2147483647, 4294967295",
            "Test %f: -65504.000000, -1.234567, 0.000000, 1.234567, 65504.000000",
            "Test %f: 0.000054",
            # "Test %f: -340282346638528859811704183484516925440.000000, 340282346638528859811704183484516925440.000000",
        ),
        "o_x_X": (
            "Test %o: tec_cnt=04, signed expr:037777777772",
            "Test %o: 020000000000, 037777700000, 00",
            "Test %o: 077777, 017777777777, 037777777777",
            "Test %x: tec_cnt=0x4, signed expr:0xfffffffa",
            "Test %x: 0x80000000, 0xffff8000, 0x0",
            "Test %x: 0x7fff, 0x7fffffff, 0xffffffff",
            "Test %x: 0x3fa00000, 0x3fa00000",
            "Test %X: tec_cnt=0x4, signed expr:0xFFFFFFFA",
            "Test %X: 0x80000000, 0xFFFF8000, 0x0",
            "Test %X: 0x7FFF, 0x7FFFFFFF, 0xFFFFFFFF",
            "Test %X: 0x3FA00000, 0x3FA00000",
        ),
        "p_c_s": (
            "Test %c: S",
            "Test %s: This is a literal string.",
        ),
        "vector": (
            "Test %v32hhx: " + ",".join(("0x80",) * 32),
            "Test %v32hhx: " + ",".join(("0x0",) * 32),
            "Test %v32hhx: " + ",".join(("0x7f",) * 32),
            "Test %v32hhx: " + ",".join(("0xff",) * 32),
            "Test %v16hx: " + ",".join(("0x8000",) * 16),
            "Test %v16hx: " + ",".join(("0x0",) * 16),
            "Test %v16hx: " + ",".join(("0x7fff",) * 16),
            "Test %v16hx: " + ",".join(("0xffff",) * 16),
            "Test %v8hlx: " + ",".join(("0x80000000",) * 8),
            "Test %v8hlx: " + ",".join(("0x0",) * 8),
            "Test %v8hlx: " + ",".join(("0x7fffffff",) * 8),
            "Test %v8hlx: " + ",".join(("0xffffffff",) * 8),
            "Test %v8hlx: " + ",".join(("0x3fa00000",) * 8),
            "Test %v16hf: " + ",".join(("-65504.000000",) * 16),
            "Test %v16hf: " + ",".join(("0.000000",) * 16),
            "Test %v16hf: " + ",".join(("1.234375",) * 16),
            "Test %v16hf: " + ",".join(("65504.000000",) * 16),
            # "Test %v8hlf: " + ",".join(("-340282346638528859811704183484516925440.000000",) * 8),
            "Test %v8hlf: " + ",".join(("0.000000",) * 8),
            "Test %v8hlf: " + ",".join(("1.234567",) * 8),
            # "Test %v8hlf: " + ",".join(("340282346638528859811704183484516925440.000000",) * 8),
        ),
    }[printf_type]

    f_printf = gen_printf(vdtype, printf_type)
    bm = BuildManager()
    ex = bm.build(f_printf)

    py_out = np.empty(n, dtype)
    f_printf(a, py_out)

    if capfd is not None:
        py_stdout, _ = capfd.readouterr()
        for expect in expects:
            msg = f"\nExpect snippet:\n{expect}\n\nPySim Standard Output:\n{py_stdout}\n"
            assert expect in py_stdout, msg
            times = py_stdout.count(expect)
            assert times == 4, f'[PySim] The expect snippet appear "{times}" times:\n'

    npu_out = np.empty(n, dtype)
    ex(a, npu_out)

    if capfd is not None:
        npu_stdout, _ = capfd.readouterr()
        for expect in expects:
            assert expect in npu_stdout, f"\nExpect snippet:\n{expect}\n\nNPU Standard Output:\n{npu_stdout}\n"
            times = npu_stdout.count(expect)
            assert times == 4, f'[NPU] The expect snippet appear "{times}" times:\n'


if __name__ == "__main__":
    test_printf("d_i_u_f", None)
    test_printf("o_x_X", None)
    test_printf("p_c_s", None)
    test_printf("vector", None)
