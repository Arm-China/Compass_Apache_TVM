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
for i in range((size + 7) // 8):
    a = S.vload(inp0_lsram + i * 8)
    b = S.vload(inp1_lsram + i * 8)

    # Upgrade types: int8x32 --> int32x8
    a32 = S.vxtl(S.vxtl(a))
    b32 = S.vxtl(S.vxtl(b))

    # Compute: (a + zp_i0) * scale_i0
    a32 = S.vadd(a32, zp_i0, saturate=True, out_sign="s")
    a32 = S.vmul(a32, scale_i0, out_sign="s")

    # Compute: (b + zp_i1) * scale_i1
    b32 = S.vadd(b32, zp_i1, saturate=True, out_sign="s")
    b32 = S.vmul(b32, scale_i1, out_sign="s")

    # Element-wise Method Operated
    tmp_w = S.vadd(a32, b32, saturate=True)

    # Multiply with scale_o
    tmp_w = S.vmul(tmp_w, scale_o, out_sign="s")

    # Shift and do narrow convert type from 8-bit to 16-bit
    tmp_h = S.i16x16(0)
    if shift < 0:
        tmp_w = S.vsl(tmp_w, -shift)
        tmp_h = S.vnsrsr(tmp_w, 0, to_h=True)
    else:
        tmp_h = S.vnsrsr(tmp_w, shift, to_h=True)

    # Subtract zero point of outputand narrow convert from 16-bit to 8-bit
    tmp_h = S.vsub(tmp_h, zp_o, saturate=True, out_sign="s")
    tmp_b = S.vnsrsr(tmp_h, 0)

    # Pack
    out = S.vcompt(tmp_b, mask="8TFFF")

    # save
    S.vstore(out, inp0_lsram + i * 8, mask="8T24F")
```

## Upgrade type

Perform type upgrade twice on elements: first from 8-bit type to 16-bit, then to 32-bit. Variable “a” has 32 8-bit elements, and the new variable “a32” has 8 32-bit elements.

```py
# If "a" is int8x32 [0,1,2,3,...,31], then "a32" is int32x8 [0,1,2,3,4,5,6,7].
a32 = S.vxtl(S.vxtl(a))
b32 = S.vxtl(S.vxtl(b))
```

Therefore, the subsequent calculations are all on 32 bits.

## Operation of Signed and Unsigned

Compute the obeys formula: `(a + zp_i0) * scale_i0`:

- `zp_i0`: the zero point of the first input
- `scale_i0`: the scale of the first input

The zero point and scale of input0 are derived from the quantization stage. Here calculation follows the requirements and order of the quantization stage to ensure that accuracy is not lost.

```py
a32 = S.vadd(a32, zp_i0, saturate=True, out_sign="s")
a32 = S.vmul(a32, scale_i0, out_sign="s")
```

Addition uses the saturate version to avoid overflow. Because the zero point is a signed value, so the specifying sign of output is “s”. Considering that the result of addition is a signed value, thus the sign of output for multiplication is “s”.

## Downgrade Type and Pack

Downgrade type by using S.vnsrsr: the “to_h” argument of S.vnsrsr specifies whether the type of outputs is 16-bit, otherwise it is lower bit (8-bit).

```py
# Assume "tmp_h" is int16x16 [0,1,2,3,...,15]
tmp_b = S.vnsrsr(tmp_h, 0)  # "tmp_b" is int8x32 [0,0,1,0,2,0,3,0,4,...,0,15,0]
out = S.vcompt(tmp_b, mask="8TFFF")  # "out" is int8x32 [0,2,4,6,8,10,12,14,0,0,...,0]
```

Pack the interleaved result together while squeezing bubbles caused by narrow convert. The mask "8TFFF" specifies that elements in "tmp_b" need to remain, representing 8 interleaving elements visually - "0b10001000100010001000100010001" in binary.

## Complete Code
You can find the sample code in `PYTHON_PACKAGE_PATH/tvm/aipu/samples/dsl/tutorial_5_quantization_op.py`.
The placeholder `PYTHON_PACKAGE_PATH` represents the location where you install the Compass DSL
Python package, in general, it will be something like `~/.local/lib/python3.8/site-packages`.
