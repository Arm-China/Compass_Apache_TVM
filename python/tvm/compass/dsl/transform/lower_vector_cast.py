# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Lower the virtual vector cast instruction to the composite of multiple real instructions."""
from tvm import tir, ir, get_range
from .. import script as S


def _vcvt(x):
    ret_vdtype_map = {
        "int32x8": "float32x8",
        "float32x8": "int32x8",
        "float16x16": "bfloat16x16",
        "bfloat16x16": "float16x16",
    }
    return tir.call_extern(ret_vdtype_map[x.dtype], "__vcvt", x)


def _vcvtd(x, y):
    return tir.call_extern("float16x16", "__vcvtd", x, y)


def _vcvtd_tbf16(x, y):
    return tir.call_extern("bfloat16x16", "__vcvtd_tbf16", x, y)


def _reinterpret_or_nsr(x, to_dtype, saturate):
    assert isinstance(saturate, bool), 'Error in script API "S.cast".'

    if saturate is False:
        return S.reinterpret(x, to_dtype)

    shift = S.cast(0, x.dtype.with_uint())
    out_sign = "s" if to_dtype.is_int else "u"
    to_h = to_dtype.bits == 16
    return S.vnsr(x, shift, saturate=True, out_sign=out_sign, to_h=to_h)


def _reinterpret_or_clip_reinterpret(x, to_dtype, saturate):
    assert isinstance(saturate, bool), 'Error in script API "S.cast".'

    if saturate is True:
        x = S.clip(x, *get_range(to_dtype))

    return S.reinterpret(x, to_dtype)


def _get_vcvtu_func(to_dtype, tgt_version):
    vcvtue = lambda expr: tir.call_extern("float32x8", "__vcvtue_tfp32", expr)
    vcvtuo = lambda expr: tir.call_extern("float32x8", "__vcvtuo_tfp32", expr)
    if to_dtype == "int32x8":
        vcvtue = lambda expr: tir.call_extern(
            "int32x8", "convert_int8", tir.call_extern("float32x8", "__vcvtue_tfp32", expr)
        )
        vcvtuo = lambda expr: tir.call_extern(
            "int32x8", "convert_int8", tir.call_extern("float32x8", "__vcvtuo_tfp32", expr)
        )

    vcvtul = lambda expr: S.vzip(vcvtue(expr), vcvtuo(expr), part="low")
    vcvtuh = lambda expr: S.vzip(vcvtue(expr), vcvtuo(expr), part="high")
    if tgt_version in ("X3P", "X3S"):
        vcvtul = lambda expr: tir.call_extern("float32x8", "__vcvtul_tfp32", expr)
        vcvtuh = lambda expr: tir.call_extern("float32x8", "__vcvtuh_tfp32", expr)
        if to_dtype == "int32x8":
            vcvtul = lambda expr: tir.call_extern(
                "int32x8", "convert_int8", tir.call_extern("float32x8", "__vcvtul_tfp32", expr)
            )
            vcvtuh = lambda expr: tir.call_extern(
                "int32x8", "convert_int8", tir.call_extern("float32x8", "__vcvtuh_tfp32", expr)
            )
    return vcvtue, vcvtuo, vcvtul, vcvtuh


class _Mutator(tir.StmtExprMutator):
    def __init__(self, cps_info):
        super().__init__()
        self._cps_info = cps_info

    def _cast_to_narrower_with_merge(self, inputs, to_dtype, saturate):
        from_dtype, input_cnt = inputs[0].dtype, len(inputs)
        msg = f'Unsupported cast from {(from_dtype,) * input_cnt} to "{to_dtype}" with saturate '
        msg += f'"{saturate}".'

        # 1. From 16-bit integer, e.g., (i16x16, i16x16) -> i8x32, (u16x16, u16x16) -> i8x32.
        if from_dtype in ("int16x16", "uint16x16"):
            assert to_dtype in ("int8x32", "uint8x32") and input_cnt == 2, msg
            inputs = tuple(_reinterpret_or_nsr(x, to_dtype, saturate) for x in inputs)
            return S.vconcat((inputs[0], inputs[1]), "even")

        # 2. From 32-bit integer.
        if from_dtype in ("int32x8", "uint32x8"):
            # 2.1 To 8-bit integer, e.g., (i32x8, i32x8, i32x8, i32x8) -> i8x32.
            if to_dtype in ("int8x32", "uint8x32"):
                inputs = tuple(_reinterpret_or_nsr(x, to_dtype, saturate) for x in inputs)
                low_16x16 = S.vconcat((inputs[0], inputs[1]), "even")

                if input_cnt == 2:
                    useless = low_16x16
                    return S.vconcat((low_16x16, useless), "even")
                if input_cnt == 3:
                    useless = inputs[2]
                    return S.vconcat((low_16x16, S.vconcat((inputs[2], useless), "even")), "even")

                assert input_cnt == 4, msg
                return S.vconcat((low_16x16, S.vconcat((inputs[2], inputs[3]), "even")), "even")

            # 2.2 To 16-bit integer, e.g., (i32x8, i32x8) -> u16x16, (u32x8, u32x8) -> i16x16.
            if to_dtype in ("int16x16", "uint16x16"):
                assert input_cnt == 2, msg
                inputs = tuple(_reinterpret_or_nsr(x, to_dtype, saturate) for x in inputs)
                return S.vconcat((inputs[0], inputs[1]), "even")

            # 2.3 To 16-bit float or bfloat, e.g., (i32x8, i32x8) -> fp16x16
            is_floating16 = to_dtype.endswith("float16x16")
            assert is_floating16 and from_dtype == "int32x8" and input_cnt == 2, msg
            vcvtd = _vcvtd if to_dtype == "float16x16" else _vcvtd_tbf16
            return vcvtd(inputs[0], inputs[1])

        # 3. From 16-bit float, e.g., (fp16x16, fp16x16) -> u8x32.
        # They are forbidden by script API "S.cast", because there isn't any hardware conversion
        # instructions between float16x16 and narrower bits integer.

        # 4. From 32-bit float.
        assert from_dtype == "float32x8", msg
        # 4.1 To 8-bit or 16-bit integer, e.g., (fp32x8, fp32x8, fp32x8, fp32x8) -> i8x32,
        #     (fp32x8, fp32x8) -> u16x16, (fp32x8, fp32x8, fp32x8) -> i8x32.
        if to_dtype in ("int8x32", "uint8x32", "int16x16", "uint16x16"):
            assert saturate is None, msg
            inputs = tuple(S.cast(x, "int32x8", saturate=True) for x in inputs)
            return self.visit(S.cast(inputs, to_dtype, saturate=False))

        # 4.2 To 16-bit float or bfloat, e.g., (fp32x8, fp32x8) -> fp16x16.
        assert to_dtype.endswith("float16x16") and input_cnt == 2, msg
        vcvtd = _vcvtd if to_dtype == "float16x16" else _vcvtd_tbf16
        return vcvtd(inputs[0], inputs[1])

    def _cast_from_i8_or_u8(self, x, from_dtype, to_dtype, part, saturate, msg):
        # 0. Supported conversion map of 8-bit integer.
        # 0.1 i8x32 -> u8x32   : Yes           u8x32 -> i8x32   : Yes
        # 0.2 i8x32 -> i16x16  : Yes           u8x32 -> i16x16  : Yes
        # 0.3 i8x32 -> u16x16  : Yes           u8x32 -> u16x16  : Yes
        # 0.4 i8x32 -> i32x8   : Yes           u8x32 -> i32x8   : Yes
        # 0.5 i8x32 -> u32x8   : Yes           u8x32 -> u32x8   : Yes
        # 0.6 i8x32 -> fp16x16 : No            u8x32 -> fp16x16 : No
        # 0.7 i8x32 -> fp32x8  : Yes           u8x32 -> fp32x8  : Yes
        # 0.8 i8x32 -> bf16x16 : No            u8x32 -> bf16x16 : No

        # 1. To different sign 8-bit integer, e.g., i8x32 -> u8x32.
        if to_dtype in ("int8x32", "uint8x32"):
            return _reinterpret_or_clip_reinterpret(x, to_dtype, saturate)

        same_sign_16bit_dtype = "int16x16" if from_dtype == "int8x32" else "uint16x16"
        diff_sign_16bit_dtype = "uint16x16" if from_dtype == "int8x32" else "int16x16"

        # 2. To same sign 16-bit integer, e.g., i8x32 -> i16x16.
        if to_dtype == same_sign_16bit_dtype:
            if part == "low":
                return S.vxtl(x)
            if part == "high":
                return S.vxth(x)

            useless = x
            if part == "even":
                return S.vxtl(S.vconcat((x, useless), "even"))
            assert part == "odd", msg
            return S.vxtl(S.vconcat((x, useless), "odd"))

        # 3. To different sign 16-bit integer, e.g., i8x32 -> u16x16.
        if to_dtype == diff_sign_16bit_dtype:
            x_same_sign_16x16 = S.cast(x, same_sign_16bit_dtype, part)
            return self.visit(S.cast(x_same_sign_16x16, to_dtype, saturate=saturate))

        same_sign_32bit_dtype = "int32x8" if from_dtype == "int8x32" else "uint32x8"
        diff_sign_32bit_dtype = "uint32x8" if from_dtype == "int8x32" else "int32x8"

        # 4. To same sign 32-bit integer, e.g., i8x32 -> i32x8.
        if to_dtype == same_sign_32bit_dtype:
            if part == "ll":
                return S.vxtl(S.vxtl(x))
            if part == "lh":
                return S.vxth(S.vxtl(x))
            if part == "hl":
                return S.vxtl(S.vxth(x))
            assert part == "hh", msg
            return S.vxth(S.vxth(x))

        # 5. To different sign 32-bit integer, e.g., i8x32 -> u32x8.
        if to_dtype == diff_sign_32bit_dtype:
            x_same_sign_32x8 = S.cast(x, same_sign_32bit_dtype, part)
            return self.visit(S.cast(x_same_sign_32x8, to_dtype, saturate=saturate))

        # 6. To 32-bit float, e.g., u8x32 -> fp32x8.
        assert to_dtype == "float32x8", msg
        x_i32x8 = S.cast(x, "int32x8", part)
        return self.visit(S.cast(x_i32x8, to_dtype))

    def _cast_from_i16_or_u16(self, x, from_dtype, to_dtype, part, saturate, msg):
        # 0. Supported conversion map of 16-bit integer.
        # 0.1 i16x16 -> i8x32   : Yes           u16x16 -> i8x32   : Yes
        # 0.2 i16x16 -> u8x32   : Yes           u16x16 -> u8x32   : Yes
        # 0.3 i16x16 -> u16x16  : Yes           u16x16 -> i16x16  : Yes
        # 0.4 i16x16 -> i32x8   : Yes           u16x16 -> i32x8   : Yes
        # 0.5 i16x16 -> u32x8   : Yes           u16x16 -> u32x8   : Yes
        # 0.6 i16x16 -> fp16x16 : No            u16x16 -> fp16x16 : No
        # 0.7 i16x16 -> fp32x8  : Yes           u16x16 -> fp32x8  : Yes
        # 0.8 i16x16 -> bf16x16 : No            u16x16 -> bf16x16 : No

        # 1. To 8-bit integer, e.g., i16x16 -> i8x32, u16x16 -> i8x32.
        if to_dtype in ("int8x32", "uint8x32"):
            assert part == "all", msg
            useless = x_8x32 = _reinterpret_or_nsr(x, to_dtype, saturate)
            return S.vconcat((x_8x32, useless), "even")

        # 2. To different sign 16-bit integer, e.g., i16x16 -> u16x16.
        if to_dtype in ("int16x16", "uint16x16"):
            return _reinterpret_or_clip_reinterpret(x, to_dtype, saturate)

        same_sign_32bit_dtype = "int32x8" if from_dtype == "int16x16" else "uint32x8"
        diff_sign_32bit_dtype = "uint32x8" if from_dtype == "int16x16" else "int32x8"

        # 3. To same sign 32-bit integer, e.g., i16x16 -> i32x8.
        if to_dtype == same_sign_32bit_dtype:
            if part == "low":
                return S.vxtl(x)
            if part == "high":
                return S.vxth(x)

            useless = x
            if part == "even":
                return S.vxtl(S.vconcat((x, useless), "even"))
            assert part == "odd", msg
            return S.vxtl(S.vconcat((x, useless), "odd"))

        # 4. To different sign 32-bit integer, e.g., i16x16 -> u32x8.
        if to_dtype == diff_sign_32bit_dtype:
            x_same_sign_32x8 = S.cast(x, same_sign_32bit_dtype, part)
            return self.visit(S.cast(x_same_sign_32x8, to_dtype, saturate=saturate))

        # 5. To 32-bit float, e.g., u16x16 -> fp32x8.
        assert to_dtype == "float32x8", msg
        x_i32x8 = S.cast(x, "int32x8", part)
        return self.visit(S.cast(x_i32x8, to_dtype))

    def _cast_from_i32_or_u32(self, x, from_dtype, to_dtype, part, saturate, msg):
        # 0. Supported conversion map of 32-bit integer.
        # 0.1 i32x8 -> i8x32   : Yes           u32x8 -> i8x32   : Yes
        # 0.2 i32x8 -> u8x32   : Yes           u32x8 -> u8x32   : Yes
        # 0.3 i32x8 -> i16x16  : Yes           u32x8 -> i16x16  : Yes
        # 0.4 i32x8 -> u16x16  : Yes           u32x8 -> u16x16  : Yes
        # 0.5 i32x8 -> u32x8   : Yes           u32x8 -> i32x8   : Yes
        # 0.6 i32x8 -> fp16x16 : Yes           u32x8 -> fp16x16 : No
        # 0.7 i32x8 -> fp32x8  : Yes           u32x8 -> fp32x8  : No
        # 0.8 i32x8 -> bf16x16 : Yes           u32x8 -> bf16x16 : No

        # 1. To 8-bit integer, e.g., i32x8 -> i8x32, u32x8 -> i8x32.
        if to_dtype in ("int8x32", "uint8x32"):
            assert part == "all", msg
            x_8x32 = _reinterpret_or_nsr(x, to_dtype, saturate)
            return S.vcompt(x_8x32, "8TFFF")

        # 2. To 16-bit integer, e.g., i32x8 -> i16x16, u32x8 -> i16x16.
        if to_dtype in ("int16x16", "uint16x16"):
            assert part == "all", msg
            useless = x_16x16 = _reinterpret_or_nsr(x, to_dtype, saturate)
            return S.vconcat((x_16x16, useless), "even")

        # 3. To different sign 32-bit integer, e.g., i32x8 -> u32x8.
        if to_dtype in ("int32x8", "uint32x8"):
            return _reinterpret_or_clip_reinterpret(x, to_dtype, saturate)

        # 4. To 16-bit float or bfloat, e.g., i32x8 -> fp16x16.
        if to_dtype.endswith("float16x16"):
            assert from_dtype == "int32x8" and part == "all", msg
            useless = x
            vcvtd = _vcvtd if to_dtype == "float16x16" else _vcvtd_tbf16
            return vcvtd(x, useless)

        # 5. To 32-bit float, e.g., i32x8 -> fp32x8.
        assert to_dtype == "float32x8" and from_dtype == "int32x8", msg
        return _vcvt(x)

    def _cast_from_fp16(self, x, to_dtype, part, saturate, msg):
        # 0. Supported conversion map of 16-bit float.
        # 0.1 fp16x16 -> i8x32   : No
        # 0.2 fp16x16 -> u8x32   : No
        # 0.3 fp16x16 -> i16x16  : No
        # 0.4 fp16x16 -> u16x16  : No
        # 0.5 fp16x16 -> i32x8   : Yes
        # 0.6 fp16x16 -> u32x8   : Yes
        # 0.7 fp16x16 -> fp32x8  : Yes
        # 0.8 fp16x16 -> bf16x16 : Yes

        # 1. To 16-bit bfloat, e.g., fp16x16 -> bf16x16.
        if to_dtype == "bfloat16x16":
            return _vcvt(x)

        # 2. To 32-bit sign integer or float, e.g., fp16x16 -> i32x8, fp16x16 -> fp32x8.
        if to_dtype in ("int32x8", "float32x8"):
            if to_dtype == "int32x8":
                assert saturate is True, msg

            vcvtue, vcvtuo, vcvtul, vcvtuh = _get_vcvtu_func(to_dtype, self._cps_info.version)
            if part == "low":
                return vcvtul(x)
            if part == "high":
                return vcvtuh(x)
            if part == "even":
                return vcvtue(x)
            assert part == "odd", msg
            return vcvtuo(x)

        # 3. To 32-bit unsigned integer, e.g., fp16x16 -> u32x8.
        assert to_dtype == "uint32x8" and saturate is None, msg
        x_i32x8 = S.cast(x, "int32x8", part, saturate=True)
        return self.visit(S.cast(x_i32x8, to_dtype, saturate=False))

    def _cast_from_bf16(self, x, to_dtype, part, saturate, msg):
        # 0. Supported conversion map of 16-bit bfloat.
        # 0.1 bf16x16 -> i8x32   : No
        # 0.2 bf16x16 -> u8x32   : No
        # 0.3 bf16x16 -> i16x16  : No
        # 0.4 bf16x16 -> u16x16  : No
        # 0.5 bf16x16 -> i32x8   : Yes
        # 0.6 bf16x16 -> u32x8   : No
        # 0.7 bf16x16 -> fp16x16 : Yes
        # 0.8 bf16x16 -> fp32x8  : Yes

        # 1. To 16-bit float, e.g., bf16x16 -> fp16x16.
        if to_dtype == "float16x16":
            return _vcvt(x)

        # 2. To 32-bit sign integer or float, e.g., bf16x16 -> i32x8, bf16x16 -> fp32x8.
        assert to_dtype in ("int32x8", "float32x8")
        if to_dtype == "int32x8":
            assert saturate is True, msg

        vcvtue, vcvtuo, vcvtul, vcvtuh = _get_vcvtu_func(to_dtype, self._cps_info.version)
        if part == "low":
            return vcvtul(x)
        if part == "high":
            return vcvtuh(x)
        if part == "even":
            return vcvtue(x)
        assert part == "odd", msg
        return vcvtuo(x)

    def _cast_from_fp32(self, x, to_dtype, part, saturate, msg):
        # 0. Supported conversion map of 32-bit float.
        # 0.1 fp32x8 -> i8x32   : Yes
        # 0.2 fp32x8 -> u8x32   : Yes
        # 0.3 fp32x8 -> i16x16  : Yes
        # 0.4 fp32x8 -> u16x16  : Yes
        # 0.5 fp32x8 -> i32x8   : Yes
        # 0.6 fp32x8 -> u32x8   : No
        # 0.7 fp32x8 -> fp16x16 : Yes
        # 0.8 fp32x8 -> bf16x16 : Yes

        # 1. To 8-bit or 16-bit integer, e.g., fp32x8 -> i8x32, fp32x8 -> u16x16.
        if to_dtype in ("int8x32", "uint8x32", "int16x16", "uint16x16"):
            assert saturate is None, msg
            x_i32x8 = S.cast(x, "int32x8", saturate=True)
            return self.visit(S.cast(x_i32x8, to_dtype, part, saturate=False))

        # 2. To 32-bit integer, e.g., fp32x8 -> i32x8.
        if to_dtype == "int32x8":
            assert saturate is True, msg
            return tir.call_extern("int32x8", "convert_int8", x)

        # 3. To 16-bit float or bfloat, e.g., fp32x8 -> fp16x16, fp32x8 -> bf16x16.
        assert to_dtype.endswith("float16x16") and part == "all", msg
        useless = x
        vcvtd = _vcvtd if to_dtype == "float16x16" else _vcvtd_tbf16
        return vcvtd(x, useless)

    def _mutate_vcast(self, call):
        _, part, saturate, *inputs = call.args
        saturate = None if isinstance(saturate, tir.StringImm) else bool(saturate)
        to_dtype = call.dtype

        # 1. To narrower bits with merge, e.g., (fp32x8, fp32x8) -> fp16x16,
        #    (i32x8, i32x8, i32x8) -> i8x32, (i16x16, i16x16) -> i8x32.
        if len(inputs) > 1:
            assert part == "all", f'Unsupported cast that merge to "{to_dtype}" with part "{part}".'
            return self._cast_to_narrower_with_merge(inputs, to_dtype, saturate)

        # From here, the input of cast must be a single vector expression.
        x = inputs[0]
        from_dtype = x.dtype
        msg = f'Unsupported cast from "{from_dtype}" to "{to_dtype}" with part "{part}" and '
        msg += f'saturate "{saturate}".'

        # 2. Redundant cast, e.g., i8x32 -> i8x32.
        if from_dtype == to_dtype:
            return x

        # 3. From 8-bit integer, e.g., i8x32 -> i16x16, u8x32 -> fp32x8.
        if from_dtype.bits == 8 and from_dtype.is_integer:
            return self._cast_from_i8_or_u8(x, from_dtype, to_dtype, part, saturate, msg)

        # 4. From 16-bit integer, e.g., i16x16 -> i8x32, u16x16 -> fp32x8.
        if from_dtype.bits == 16 and from_dtype.is_integer:
            return self._cast_from_i16_or_u16(x, from_dtype, to_dtype, part, saturate, msg)

        # 5. From 32-bit integer, e.g., i32x8 -> i8x32, u32x8 -> i16x16, i32x8 -> fp32x8.
        if from_dtype.bits == 32 and from_dtype.is_integer:
            return self._cast_from_i32_or_u32(x, from_dtype, to_dtype, part, saturate, msg)

        # 6. From 16-bit float, e.g., fp16x16 -> i32x8, fp16x16 -> fp32x8.
        if from_dtype.is_float16:
            return self._cast_from_fp16(x, to_dtype, part, saturate, msg)

        # 7. From 16-bit bfloat, e.g., bf16x16 -> i32x8, bf16x16 -> fp32x8.
        if from_dtype.is_bfloat16:
            return self._cast_from_bf16(x, to_dtype, part, saturate, msg)

        # 8. From 32-bit float, e.g., fp32x8 -> i32x8, fp32x8 -> u8x32, fp32x8 -> fp16x16.
        assert from_dtype.is_float32, msg
        return self._cast_from_fp32(x, to_dtype, part, saturate, msg)

    def visit_call(self, call):
        ret = super().visit_call(call)

        if ret.op != ir.Op.get("tir.call_extern"):
            return ret

        func_name = ret.args[0].value
        if func_name == "vcast":
            return self._mutate_vcast(ret)

        return ret


@tir.transform.prim_func_pass(opt_level=0)
class LowerVectorCast:
    """Lower the virtual vector cast instruction to the composite of multiple real instructions."""

    def __init__(self, cps_info):
        self._cps_info = cps_info

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        return func.with_body(_Mutator(self._cps_info).visit(func.body), span=func.span)
