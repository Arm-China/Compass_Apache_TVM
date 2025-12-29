<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.-->

# How to Use Mask
This section describes how to use mask for vectorized programming. In vectorized programming, you can use mask if you want to compute specific elements in vector.

Valid types of mask are as follows:

- None.
- List or tuple or numpy array of bool.
- String representation.
- Result of vector comparison instructions.

## None

The default value of mask is None, which means all elements to be selected. The following code shows a basic example of element-wise add in vectors without mask.

```py
def vadd(a: S.ptr("i32x8", "global"), b: S.ptr("i32x8", "global"), c: S.ptr("i32x8", "global")):
    c[0] = S.vadd(a[0], b[0])
```

## List, Tuple or Numpy Array of Boolean

The list, tuple or numpy array of boolean values can be used as a mask. The elements will be selected if the corresponding value in mask is True.

```py
def vadd(a: S.ptr("i32x8", "global"), b: S.ptr("i32x8", "global"), c: S.ptr("i32x8", "global")):
    c[0] = S.vadd(a[0], b[0], mask=[True, False, True, True, False, True, False, False])
```

If you want to save boolean array to a variable and pass the variable to a built-in API, you need to create the variable by S.const_mask. The code is as follows:

```py
def vadd(a: S.ptr("i32x8", "global"), b: S.ptr("i32x8", "global"), c: S.ptr("i32x8", "global")):
    mask = S.const_mask([True, False, True, True, False, True, False, False])
    c[0] = S.vadd(a[0], b[0], mask=mask)
```

## String Representation

The string representation of mask can write all the "T" (true), "F" (false) manually such as "FTFTFTFT". For simplicity, "FTFTFTFT" can also be "4FT". The rule is based on the repeat times number and repeat pattern.

- Repeat Times Number: Specify the number of times that a pattern repeats. This is to avoid ambiguity. For example, "4FT" could be misinterpreted as either four "F"s followed by a single "T" or as a pattern of "FT" repeated four times. To clarify, the format is "repeat number + pattern". If you intend to represent four "F"s followed by a single "T", the pattern must be broken up to avoid confusion, such as "4F1T".
- Repeat Pattern: The pattern can consist of a single character representing a true or false value, or it can be a sequence of characters for more complex patterns. For example, "2FFFT3T" indicates that the pattern "FFFT" is repeated twice, followed by another pattern "T" repeated 3 times.

Tips:

- The first repeat times number in string representation can be ignored when it equals 1, such as "F4T" (equals "1F4T").
- Auto padding mask when high part elements of mask are not provided. Usually, we write full lanes of mask, such as "vadd(va, vb, mask='3T5F')", but you can write "vadd(va, va, mask='3T')" instead, because the high part of mask will be filled with "F"(s) automatically according to lanes of vector data (here is "va").
- If the string is too long, you can use underscores to separate it, such as "TTFT_TFTT", "4T_FT3F_TT".

Here are mask examples with string representation:

```py
def lanes_eq_32(a: S.ptr("i8x32", "global"), out: S.ptr("i8x32", "global")):
    out[0] = S.vadd(a[0], 1, mask="16T16F")  # Can also be "16T".
```

```py
def lanes_eq_16(a: S.ptr("i16x16", "global"), b: S.ptr("i16x16", "global"), out: S.ptr("i16x16", "global")):
    out[0] = S.vadd(a[0], b[0], mask="4TF8F")  # Can also be "4TF".
```

```py
def lanes_eq_8(a: S.ptr("i32x8", "global"), b: S.i32, out: S.ptr("i32x8", "global")):
    out[0] = S.vadd(a[0], b, mask="1TTFT4F")  # Can also be "TTFT4F" or "TTFT".
```

In addition, the sequence of string representation from the leftmost element to the rightmost element matches the corresponding elements from the lowest one to the highest one of vector data, i.e., the leftmost one is the lowest one.

You also can save the string representation value to a variable, then use it as a mask as follows:

```py
def vadd(a: S.ptr("i8x32", "global"), b: S.ptr("i8x32", "global"), c: S.ptr("i8x32", "global")):
    mask = S.const_mask("8TFFF")
    c[0] = S.vadd(a[0], b[0], mask=mask)
```

## Result of Vector Comparison Instructions

Mask can also be a return value of some instructions, such as comparison instructions. The following code shows how to perform element-wise add on elements a_i greater than b_i.

```py
def vadd(a: S.ptr("i32x8", "global"), b: S.ptr("i32x8", "global"), c: S.ptr("i32x8", "global")):
    mask = S.vcgt(a[0], b[0])
    c[0] = S.vadd(a[0], b[0], mask=mask)
```
