<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.-->

# Static and Dynamic Kernel

There are two kinds of kernels, depending on whether the shapes of tensors are known at compile-time or not.

- **Static kernel**: Tensor shape is known and fixed as constant values at compile-time.
- **Dynamic kernel**: Tensor shape is variable and the exact shape values are not determined until runtime.


## Static Kernel

### Static Kernel Example

To declare pointer parameters for a static kernel, there are typically two main approaches:

1. Use `S.ptr + S.match_buffer` with specific shape and data type for multi-dimension data.
    ```py
    @S.prim_func
    def static_add_1(A: S.ptr("i32", "global")):
        a = S.match_buffer(A, shape=(4, 8))

        for i in range(4):
            for j in range(8):
                a[i, j] = a[i, j] + 2
    ```
     The generated c_code is:
    ```c
    __kernel void static_add_1(__global int* A) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 8; ++j) {
                A[i * 8 + j] = (A[i * 8 + j] + 2);
            }
        }
    }
    ```
2. Use `S.ptr` with specific data type and scope for 1-Dimension data.
    ```py
    n = 32 # value known at compile-time

    @S.prim_func
    def static_add_2(A: S.ptr("i32", "global")):
        for i in range(n):
            A[i] = A[i] + 2
    ```
     The generated c_code is:
    ```c
    __kernel void static_add_2(__global int* A) {
        for (int i = 0; i < 32; ++i) {
            A[i] = (A[i] + 2);
        }
    }
    ```
3. See [tutorials/1_static_add](../getting_started/tutorials/1_static_add.md).

### Static Kernel Plugin

The static kernel plugin is the same as the plugin introduced in [How to Write the Operator Plugin](../how_to_guides/how_to_write_the_operator_plugin.md). The only difference is to set the **check_params** section to enable only specific shape.

```{note}
**Why & When to Use Static Kernel**

1. Static kernel can leverage shape information known at compile-time to optimize the implementation for better performance. So you can use the static kernel when you have performance requirements with operator shape known at compile-time.
2. Static kernel can also serve as a valuable debugging tool for verifying the correctness of a dynamic kernel. By comparing the intermediate results with fixed, known input shapes, developers can more easily identify and resolve issues in the dynamic kernel’s implementation. So you can use the static kernel for debugging purpose.
```
## Dynamic Kernel
### Dynamic Kernel Examples

For a dynamic kernel, shape of tensors can be viewed as a variable that is passed in as a kernel argument. We use `S.ptr(dtype)` to declare the tensor pointer at kernel arguments.

1. Use `S.ptr(dtype, scope)`:  for 1-Dimension data, you can skip the match_buffer step.
    ```py
    # for 1-dimension data, without match_buffer is OK
    @S.prim_func
    def dynamic_add(A: S.ptr("int32", "global"), n: S.int32):
        for i in range(n):
            A[i] = A[i] + 2
    ```
    The generated c_code is:
    ```c
    __kernel void dynamic_add(__global int* A, int n) {
    for (int i = 0; i < n; ++i) {
        A[i] = (A[i] + 2);
    }
    }
    ```
2. Use `S.ptr(dtype, scope) + S.match_buffer(with shape)`: for both one-Dimension and multi-Dimension data.
    ```py
    # for 2-dimension data, use S.ptr(dtype, scope) + S.match_buffer(with shape)
    @S.prim_func
    def matrix_transpose(A: S.ptr("int8", "global"), B: S.ptr("int8", "global"), h: S.int32, w: S.int32):
        a = S.match_buffer(A, shape=(h, w))
        b = S.match_buffer(B, shape=(w, h))

        for ih, iw in S.grid(h, w):
            b[iw, ih] = a[ih, iw]
    ```

3. See [tutorials/2_dynamic_add](../getting_started/tutorials/2_dynamic_add.md).

### Dynamic Kernel Plugin

The dynamic kernel plugin is the same as the plugin introduced in [How to Write the Operator Plugin](../how_to_guides/how_to_write_the_operator_plugin.md).


```{note}
**Why & When to Use Dynamic Kernel**

The benefit of a dynamic kernel is its flexibility to accommodate tensors with unknown or changing shapes at runtime. So we use the dynamic kernel to enable a single kernel to handle a wide range of input sizes and dimensions, providing a more generic solution that is not constrained by fixed dimensions.

```
