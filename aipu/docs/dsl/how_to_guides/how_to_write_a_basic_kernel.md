<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.-->

# How to Write a Basic Kernel

Learn how to write a basic kernel.
- Basic elements in a kernel (primfunc)
- Declare function parameters
- Loop iterations
- A complete kernel demo


## Basic Elements in a Kernel (primfunc)
A kernel is a function that performs a specific computation on a set of input data.

In Compass DSL, we write a `primfunc` to implement a kernel.

For a typical function in C, the key elements are:
- Function parameters
- Function body

Here are the typical elements in a `primfunc`:
- Function parameters
- Function body
  - **loop nests**
  - **computations**

For more information, see [TVM Doc: tensor program abstraction](https://mlc.ai/chapter_tensor_program/tensor_program.html#tensor-program-abstraction).

In Compass DSL, all primfunc should be decorated with `@S.prim_func`.

## Declare Function Parameters

We use `S.ptr(dtype)` to declare pointer parameters.
- `S.ptr(dtype) + S.match_buffer(with shape)`: For both one-dimension and multi-dimension data.
- `S.ptr(dtype)`:  For 1-Dimension data, you can skip the match_buffer step.


```py
# for 1-dimension data, without match_buffer is OK
@S.prim_func
def add_func1(A: S.ptr("int32", "global"), n:S.int32):
    for i in range(n):
        A[i] = A[i] + 2
```
The generated c_code is:
```c
__kernel void add_func1(__global int* A, int n) {
  for (int i = 0; i < n; ++i) {
    A[i] = (A[i] + 2);
  }
}
```

```py
# for 2-dimension data, use S.ptr(dtype) + S.match_buffer(with shape)
@S.prim_func
def matrix_transpose(A: S.ptr("int8", "global"), B: S.ptr("int8", "global"), h: S.int32, w: S.int32):
    a = S.match_buffer(A, shape=(h, w))
    b = S.match_buffer(B, shape=(w, h))

    for ih, iw in S.grid(h, w):
        b[iw, ih] = a[ih, iw]
```
## Loop Iterations
1. Use python `range` directly.

    ```py
    @S.prim_func
    def matrix_transpose(A: S.ptr("int8", "global"), B: S.ptr("int8", "global")):
        a = S.match_buffer(A, shape=(2, 3))
        b = S.match_buffer(B, shape=(3, 2))

        for h in range(3):
            for w in range(2):
                b[h, w] = a[w, h]
    ```
2. Use `S.grid` syntactic sugar in TensorIR to write multiple nested iterators.

    ```py
    @S.prim_func
    def matrix_transpose(A: S.ptr("int8", "global"), B: S.ptr("int8", "global")):
        a = S.match_buffer(A, shape=(2, 3))
        b = S.match_buffer(B, shape=(3, 2))

        for h, w in S.grid(3, 2):
            b[h, w] = a[w, h]
    ```

## A Complete Kernel Demo
Here is an example with the matrix transpose kernel.

```py
import numpy as np
from tvm import aipu
from tvm.aipu import script as S

@S.prim_func
def matrix_transpose(A: S.ptr("int8", "global"), B: S.ptr("int8", "global"), h: S.int32, w: S.int32):
    a = S.match_buffer(A, shape=(h, w))
    b = S.match_buffer(B, shape=(w, h))

    for ih, iw in S.grid(h, w):
        b[iw, ih] = a[ih, iw]


def test_func():
    bm = aipu.tir.BuildManager()
    ex = bm.build(matrix_transpose)
    print(ex.c_code)

    h = 2
    w = 3
    a = np.array(list(range(h * w)), dtype="int8")
    b = np.zeros((h * w,), dtype="int8")
    ex(a, b, h, w)
    print(a, b)

if __name__ == "__main__":
    test_func()
```
The generated c code will be:
```c
__kernel void matrix_transpose(__global char* a, __global char* b, int h, int w) {
  for (int ih = 0; ih < h; ++ih) {
    for (int iw = 0; iw < w; ++iw) {
      b[((iw * h) + ih)] = a[((ih * w) + iw)];
    }
  }
}
```
The test input and output data is:
```
a = [0 1 2 3 4 5]
b = [0 3 1 4 2 5]
```

For more information about how to use the DSL language to write the function body computation, see [Language Basics](../language_basics/index.rst).
