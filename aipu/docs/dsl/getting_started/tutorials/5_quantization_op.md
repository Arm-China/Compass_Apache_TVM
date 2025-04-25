<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.-->

# Tutorial 5: Quantization Operator

Based on previous Tutorial 1 Static Add and Tutorial 2 Dynamic Add, in this tutorial, you will write a quantization vector addition with dynamic kernel using Compass DSL. You will learn about:

* How to upgrade low-level wide types to high-precision types.
* How to determine the type of output when two operands are signed and unsigned in an operation.
* How to convert high width types to low width types.

## Inputs & Outputs

- Inputs:
  - Tensor(a,shape=(n,), dtype="int8")
  - Tensor(b,shape=(n,), dtype="int8")
- Output:
  - Tensor(c,shape=(n,), dtype="int8")
- Others:
  - zero point:
    - input0 zp: int16
    - input1 zp: int16
    - output zp: int16
  - scale
    - input0 scale: uint16
    - input1 scale: uint16
    - output scale: uint8
  - shift: int8

So we can write the primfunc like this:

```py
dtype = "int8"

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
    # func body
    ...
```

Quantization operator implementation is tight with the quantization stage. This tutorial concentrates on operator implementation but not quantization algorithm.

## Calculation

The following is the main calculation process.

```py
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
        a32_add = a32 + zp_i0
        a32_mul = S.vmul(a32_add, scale_i0, out_sign="s")

        # Compute: (b + zp_i1) * scale_i1.
        b32_add = b32 + zp_i1
        b32_mul = S.vmul(b32_add, scale_i1, out_sign="s")

        # Element-wise with method=ADD.
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
```

## Upgrade type

Load 16 8-bit elements on "inp0_lsram" and "inp1_lsram", and then cast them to 32-bit. Variables "a32" and "b32" both has 16 32-bit elements.

```py
a32 = S.cast(inp0_lsram[idx_base : idx_base + 16], "int32")
b32 = S.cast(inp1_lsram[idx_base : idx_base + 16], "int32")
```

Therefore, the subsequent calculations are all on 32 bits.

## Operation of Signed and Unsigned

Compute the obeys formula: `(a + zp_i0) * scale_i0`:

- `zp_i0`: the zero point of the first input
- `scale_i0`: the scale of the first input

The zero point and scale of input0 are derived from the quantization stage. Here calculation follows the requirements and order of the quantization stage to ensure that accuracy is not lost.

The addition here does not use saturation operation. This is because the value contained in the 32-bit type "a32" is 8-bit, and the zero point "zp_i0" is 8-bit. The final result will not exceed 9 bits.

```py
a32_add = a32 + zp_i0
a32_mul = S.vmul(a32_add, scale_i0, out_sign="s")
```

Because the zero point is a signed value, so the specifying sign of output is "s". Considering that the result of addition is a signed value, thus the sign of output for multiplication is "s".

## Downgrade Type and Pack

Downgrade type from 16 16-bit elements to 8-bit, and then store to "inp0_lsram".

```py
inp0_lsram[idx_base : idx_base + 16] = S.cast(tmp_h, dtype, saturate=True)
```

## Complete Code
You can find the sample code in `PYTHON_PACKAGE_PATH/tvm/aipu/samples/dsl/tutorial_5_quantization_op.py`.
The placeholder `PYTHON_PACKAGE_PATH` represents the location where you install the Compass DSL
Python package, in general, it will be something like `~/.local/lib/python3.8/site-packages`.
