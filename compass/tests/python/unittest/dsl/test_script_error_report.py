# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import re
import numpy as np
import pytest
import logging
from contextlib import contextmanager
from tvm.compass.dsl import BuildManager, script as S, Aiff


@contextmanager
def _logger_propagate():
    logger = logging.getLogger("Compass")
    try:
        logger.propagate = True
        yield
    finally:
        logger.propagate = False


@S.prim_func
def sub_func(a: S.i32, b: S.i32x8, c: S.ptr("fp16")):
    return


def test_sub_func_vector_to_scalar(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func():
            x = S.fp16(1)
            sub_func(S.i32x8(3), S.i32x8(2), x.addr)

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = "The 1-th arg expect a scalar."
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_sub_func_scalar_to_vector(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func():
            x = S.fp16(1)
            sub_func(S.i32(3), S.i32(2), x.addr)

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = "The 2-th arg expect a vector."
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_sub_func_vector_to_pointer(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func():
            sub_func(S.i32(3), S.i32x8(2), S.i32x8(2))

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = "The 3-th arg expect a pointer."
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_sub_func_pointer_to_scalar(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func():
            x = S.fp16(1)
            sub_func(x.addr, S.i32x8(2), x.addr)

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = "The 1-th arg expect a variable, but got: pointer."
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_sub_func_mismatch_scope(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func(x: S.ptr("fp16", "lsram")):
            sub_func(S.i32(3), S.i32x8(2), x)

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = 'The scope of 3-th arg expect "private", but got: "lsram".'
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_for_iter_var_defined_outside(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func(a: S.ptr("int32x8", "global"), b: S.ptr("int32x8", "global")):
            i = 0
            for i in range(1):
                b[i] = a[i]

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = "The iter var i of the for loop has been defined in outside scope, please use another name."
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_empty_ret_type(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def sub_func():
            return 1

        @S.prim_func
        def fail_func(a: S.ptr("int32x8", "global"), b: S.ptr("int32x8", "global")):
            loop = sub_func()
            for i in range(loop):
                b[i] = a[i]

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = "This function lacks return type annotation."
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_string_var(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func(a: S.ptr("int32x8", "global"), b: S.ptr("int32x8", "global")):
            dtype = "int32x8"
            lsram = S.alloc_buffer(shape=(32,), dtype=dtype, scope="lsram")
            b[0] = lsram[0]

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = "Can't define string inside primitive function, please define it outside."
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_builtin_tuple_to_mask(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func(a: S.ptr("i32x8", "global"), b: S.ptr("i32x8", "global"), out: S.ptr("i32x8", "global")):
            mask_out = (a[0] > S.i32(151850249)), (b[0] < S.i32(0))
            out[0] = S.vsel(a[0], b[0], mask_out)

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = 'All elements of arg "mask" expect "True" or "False".'
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_builtin_pointer_to_vector(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func(a: S.ptr("i32x8", "global"), b: S.i32, out: S.ptr("i32x8", "global")):
            out[0] = S.vadd(a, b.addr)

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = 'Unexpect "int32x8*" pointer as argument.'
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_assert_neither_flexible_nor_multiple_width_vector_error(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func(a: S.ptr("int8", "global"), b: S.ptr("int8", "global")):
            b[0] = S.vxtl(S.vload(a, lanes=64))

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = 'This API expect hardware native vector types, but got: "int8x64".'
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_ptr_invalid_dtype():
    with pytest.raises(AssertionError) as exc_info:

        @S.prim_func
        def fail_func(a: S.ptr("float64", "global"), b: S.ptr("int32", "global")):
            b[0] = a[0]

        BuildManager().build(fail_func)

    exc_msg = str(exc_info.value)
    expect = r'The scalar form of arg "dtype" expect one of .*, but got: "float64".'
    matches = re.search(expect, exc_msg, re.MULTILINE)
    assert matches is not None, f"\nExpect snippet:\n{expect}\n\nException Message:\n{exc_msg}\n"


def test_as_ptr_invalid_dtype(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def fail_func(a: S.ptr("fp32", "global"), b: S.ptr("int32", "global")):
            b[0] = a.as_ptr("int4")[0]

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()

    expect = r'The scalar form of arg "dtype" expect one of .*, but got: "int4".'
    matches = re.search(expect, stderr, re.MULTILINE)
    assert matches is not None, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


@pytest.mark.parametrize(
    "fail_type", ("unexpected_arg", "missing_arg", "multiple_vals", "redundant_arg0", "redundant_arg1")
)
def test_sub_func_invalid_param(fail_type, capfd):
    @S.prim_func
    def get_val(v0: S.i8, v1: S.i32 = 20, v2: S.fp16 = 30, v3: S.fp32 = 40) -> S.i32:
        return v0 + v1 + v2 + v3

    @S.prim_func
    def func_unexpected_arg(out: S.ptr("i32", "global")):
        out[0] = get_val(11, v10=22)

    @S.prim_func  # Check empty args
    def func_missing_arg(out: S.ptr("i32", "global")):
        out[0] = get_val(v1=22, v2=33)

    @S.prim_func
    def func_multiple_vals(out: S.ptr("i32", "global")):
        out[0] = get_val(11, 22, v1=33)  # pylint: disable=redundant-keyword-arg

    @S.prim_func
    def func_redundant_arg0(out: S.ptr("i32", "global")):
        out[0] = get_val(11, 22, 33, v1=22, v2=33)  # pylint: disable=redundant-keyword-arg

    @S.prim_func
    def func_redundant_arg1(a: S.ptr("i32x8", "global"), out: S.ptr("i32x8", "global")):
        sub_func(1, a[1], a, a[2])  # pylint: disable=too-many-function-args
        out[0] = a[0]

    py_func = locals()[f"func_{fail_type}"]
    with pytest.raises(RuntimeError):
        BuildManager().build(py_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()

    expect = ""
    if fail_type == "unexpected_arg":
        expect = 'error: The function "get_val" got unexpect keyword args: "v10".'
    if fail_type == "missing_arg":
        expect = 'The function "get_val" missing "1" args: "v0".'
    if fail_type == "multiple_vals":
        expect = 'The function "get_val" got multiple values for args: "v1".'
    if fail_type == "redundant_arg0":
        expect = 'The function "get_val" expect 1 to 4 args, but got: "5".'
    if fail_type == "redundant_arg1":
        expect = 'The function "sub_func" expect 3 args, but got: "4".'
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_entry_func_invalid_param_num():
    a = np.empty(1, dtype="int32")

    @S.prim_func
    def fail_func(a: S.ptr("i32x8", "global"), out: S.ptr("i32x8", "global")):
        out[0] = a[0]

    ex = BuildManager().build(fail_func)

    with pytest.raises(AssertionError) as pysim_exc_info:
        fail_func(a)

    expect = 'The function "fail_func" expect 2 args, but got: "1".'
    exc_msg = str(pysim_exc_info.value)
    assert expect in exc_msg, f"\nExpect snippet:\n{expect}\n\nException Message:\n{exc_msg}\n"

    with pytest.raises(AssertionError) as npu_exc_info:
        ex(a)

    exc_msg = str(npu_exc_info.value)
    assert expect in exc_msg, f"\nExpect snippet:\n{expect}\n\nException Message:\n{exc_msg}\n"


def test_mismatch_entry_ptr_dtype(caplog):
    a = np.empty(1, dtype="float32")
    b = np.empty(1, dtype="float32")

    @S.prim_func
    def fail_func(a: S.ptr("fp32", "global"), b: S.ptr("int32", "global")):
        b[0] = 1

    with _logger_propagate():
        ex = BuildManager().build(fail_func)
        fail_func(a, b)
        ex(a, b)

    if caplog is None:
        return
    expect = 'WARN.* The 2-th arg of function "fail_func" expect a int32 NumPy, but got: "float32".'
    matches = re.findall(expect, caplog.text)
    assert len(matches) == 2, f"\nExpect snippet:\n{expect}\n\nLog Text:\n{caplog.text}\n"


def test_mismatch_return_pointer(capfd):
    with pytest.raises(RuntimeError):

        @S.prim_func
        def sub_func(a: S.ptr("i32", "global")) -> S.ptr("i32", "global"):
            if a[0] > 0:
                ptr = S.alloc((128,), "i32")
                return ptr
            return a

        @S.prim_func
        def fail_func(a: S.ptr("i32", "global"), out: S.ptr("i32", "global")):
            a = sub_func(a)
            out[0] = a[0]

        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = 'The scope of return type expect "global", but got: "private".'
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_return_stack_pointer():
    with pytest.raises(AssertionError) as exc_info:

        @S.prim_func
        def sub_func(a: S.ptr("i32")) -> S.ptr("i32"):
            if a[0] > 0:
                ptr = S.alloc((128,), "i32")
                return ptr
            return a

        @S.prim_func
        def fail_func(out: S.ptr("i32", "global")):
            ptr = S.alloc((128,), "i32")
            ptr = sub_func(ptr)
            out[0] = ptr[0]

        BuildManager().build(fail_func)

    exc_msg = str(exc_info.value)
    expect = 'The function "sub_func" return a stack pointer. Please check it carefully.'
    assert expect in exc_msg, f"\nExpect snippet:\n{expect}\n\nException Message:\n{exc_msg}\n"


def test_use_annotation_in_python(capfd):
    def sub_func1(a):
        return S.fp32(a)

    def sub_func2(a, b: S.i32):
        x = a + 3
        return b + x

    @S.prim_func
    def fail_func(a: S.ptr("fp16x16", "global")):
        x = sub_func1(a[0])
        y = sub_func2(2, x)
        a[0] = y

    with pytest.raises(RuntimeError):
        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = 'The parameter "b" of pure Python function "sub_func2" can\'t be annotated as any type'
    expect += ' in "S" space, forget to decorate it using "S.prim_func"?'
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_discontiguous_input():
    with pytest.raises(AssertionError) as exc_info:

        @S.prim_func
        def fail_func(a: S.ptr("fp16", "global")):
            a[0] = 1

        BuildManager().build(fail_func)
        inp = np.transpose(np.zeros((4, 32), "float16"))
        fail_func(inp)

    exc_msg = str(exc_info.value)
    expect = 'The 1-th arg of function "fail_func" expect a C-style contiguous NumPy, please '
    expect += 'achieve it through "numpy.ascontiguousarray".'
    assert expect in exc_msg, f"\nExpect snippet:\n{expect}\n\nException Message:\n{exc_msg}\n"


def test_aiff_desc_dtype():
    with pytest.raises(AssertionError) as exc_info:

        @S.prim_func
        def fail_func(desc: S.ptr("float32", "global")):
            if S.get_local_id() != 0:
                return
            S.aiff(desc + 72 + 48, desc, desc + 72)

        BuildManager().build(fail_func)

    exc_msg = str(exc_info.value)
    expect = 'The descriptor parameter "desc" of function "fail_func" expect dtype "uint32/int32",'
    expect += ' but got: "float32".'
    assert expect in exc_msg, f"\nExpect snippet:\n{expect}\n\nException Message:\n{exc_msg}\n"


def test_sub_func_aiff_desc_dtype():
    with pytest.raises(AssertionError) as exc_info:

        @S.prim_func
        def call_aiff(desc: S.ptr("float32", "global")):
            S.aiff(desc + 72 + 48, desc, desc + 72)

        @S.prim_func
        def fail_func(desc: S.ptr("u32", "global")):
            if S.get_local_id() != 0:
                return
            call_aiff(desc)

        BuildManager().build(fail_func)

    exc_msg = str(exc_info.value)
    expect = 'The descriptor parameter "desc" of function "call_aiff" expect dtype "uint32/int32",'
    expect += ' but got: "float32".'
    assert expect in exc_msg, f"\nExpect snippet:\n{expect}\n\nException Message:\n{exc_msg}\n"


def test_aiff_register_property():
    with pytest.raises(AssertionError) as exc_info:

        aiff = Aiff()
        aiff.mtp.twin_ctrl.mtp_0_disen2 = 0

    exc_msg = str(exc_info.value)
    expect = 'The field "mtp_0_disen2" is not valid.'
    assert expect in exc_msg, f"\nExpect snippet:\n{expect}\n\nException Message:\n{exc_msg}\n"


def test_aiff_transposed_array_in_desc():
    with pytest.raises(AssertionError) as exc_info:
        aiff = Aiff()
        inp = np.transpose(np.zeros((4, 32), "float16"))
        aiff.ptp.oact_fp_param_addr = inp.reshape(-1)
        aiff.ptp.iact_addr = inp

    exc_msg = str(exc_info.value)
    expect = 'The field "iact_addr" receives a transposed array which could change its memory '
    expect += "layout, please use its copy instead."
    assert expect in exc_msg, f"\nExpect snippet:\n{expect}\n\nException Message:\n{exc_msg}\n"


def test_bf16_scalar_operator(capfd):
    @S.prim_func
    def fail_func(a: S.ptr("bf16", "global"), b: S.ptr("bf16", "global"), out: S.ptr("bf16", "global")):
        mask_out = a[0] == b[0]
        if mask_out:
            out[0] = a[0]
        else:
            out[0] = b[0]

    with pytest.raises(RuntimeError):
        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = "There isn't any compute instruction for bfloat16 scalar, please convert it to float32 first."
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


def test_bf16_scalar_python_call(capfd):
    binaray_add = lambda x, y: x + y

    @S.prim_func
    def fail_func(a: S.ptr("int16", "global"), b: S.ptr("bf16", "global"), out: S.ptr("bf16", "global")):
        out[0] = binaray_add(a[0], b[0])

    with pytest.raises(RuntimeError):
        BuildManager().build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = "There isn't any compute instruction for bfloat16 scalar, please convert it to float32 first."
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nStandard Error:\n{stderr}\n"


@pytest.mark.NOT_X1
@pytest.mark.NOT_X2
def test_x3p_unsupported_ir_api(capfd):
    @S.prim_func
    def fail_func(
        a: S.ptr("fp16x16", "global"),
        b: S.ptr("fp16x16", "global"),
        c: S.ptr("fp32x8", "global"),
        d: S.ptr("fp32x8", "global"),
    ):
        cc = S.vconcat((c[0], c[1]))
        S.vmma(cc.addr, a[0], b[0])
        cc0, cc1 = S.vsplit(cc)
        d[0] = cc0
        d[1] = cc1

    with pytest.raises(RuntimeError):
        BuildManager(target="X3P_1304").build(fail_func)

    if capfd is None:
        return
    _, stderr = capfd.readouterr()
    expect = 'The "vmma" does not support the target'
    assert expect in stderr, f"\nExpect snippet:\n{expect}\n\nException Message:\n{stderr}\n"


if __name__ == "__main__":
    test_sub_func_vector_to_scalar(None)
    test_sub_func_scalar_to_vector(None)
    test_sub_func_vector_to_pointer(None)
    test_sub_func_pointer_to_scalar(None)
    test_sub_func_mismatch_scope(None)
    test_for_iter_var_defined_outside(None)
    test_empty_ret_type(None)
    test_string_var(None)
    test_builtin_tuple_to_mask(None)
    test_builtin_pointer_to_vector(None)
    test_assert_neither_flexible_nor_multiple_width_vector_error(None)
    test_ptr_invalid_dtype()
    test_as_ptr_invalid_dtype(None)
    test_sub_func_invalid_param("unexpected_arg", None)
    test_entry_func_invalid_param_num()
    test_mismatch_entry_ptr_dtype(None)
    test_mismatch_return_pointer(None)
    test_return_stack_pointer()
    test_use_annotation_in_python(None)
    test_discontiguous_input()
    test_aiff_desc_dtype()
    test_sub_func_aiff_desc_dtype()
    test_aiff_register_property()
    test_aiff_transposed_array_in_desc()
    test_bf16_scalar_operator(None)
    test_bf16_scalar_python_call(None)
    test_x3p_unsupported_ir_api(None)
