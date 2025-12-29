<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.-->

# Mask
## What is Mask

Mask essentially provides conditional execution of each element operation in a vector instruction.
The vector-mask control uses a Boolean vector to control the execution of a vector instruction.
Most vector instructions in DSL provide optional mask parameters.
When the mask is enabled, vector instructions executed operate only on the vector elements whose corresponding entries in the vector-mask are True.

In DSL, the mask can come from a builtin call or an immediate value. If a mask variable comes from an immediate value, it should be a list of bool or a 32-bit unsigned int.


## Example of Mask
Here are some examples of the mask result.

```py
a = S.int32x8(-2)
b = S.abs(a, mask="2FTT1FT")
```

The following result shows elements from low to high:

- a : [-2, -2, -2, -2, -2, -2, -2, -2]
- b : [-2, 2, 2, -2, 2, 2, -2, 2]

The corresponding elements of b is an absolute value if the corresponding entries are True.

```py
a = S.int16x16(-2)
b = S.abs(a, mask=0x10110112)
```

The following result shows elements from low to high:

- a : [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]
- b : [-2, -2, 2, -2,  2, -2, -2, -2, 2, -2, 2, -2, -2, -2, 2, -2]

Note that the lower 4 bits of the mask is b’0010, but as the lane with 16, the activation bit is 0, 2, 4, 6, etc., so the lower two elements in int16x16 are not activated.

## How to Generate Mask

Other than immediate values mentioned above, DSL also provides some builtin calls to generate mask.

**S.tail_mask**

In most situations, we just want the first n element in the vector variable and abandon the last behind.

```py
a = S.tail_mask(3, 8)
```
Where, a would be a mask variable with the first 3 elements activated and the last 5 elements disactivated if used for the lane 8 vector instruction.

**Vector comparison instructions**
- vceq
- vcneq
- vcge
- vcgt
- vcle
- vclt

These comparison instructions accept two vector inputs with the same lanes and output a mask variable. The corresponding bit would be set if the comparison is True between the two input vector elements.
