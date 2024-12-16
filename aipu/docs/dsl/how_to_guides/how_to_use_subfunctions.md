<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.-->

# How to Use Subfunctions

## Introduction

In various programming contexts, especially when working with Domain-Specific Languages (DSLs), subfunctions play a crucial role.
They not only help in reducing code complexity but also enhance code readability and reusability.
However, despite the seemingly straightforward concept of subfunctions, using them correctly and efficiently requires certain skills and knowledge.

This tutorial is designed to delve into the use of subfunctions within DSLs, guiding you step by step on how to effectively utilize subfunctions to optimize your DSL code.
Whether you are new to DSLs or looking to improve the quality of your existing code, this tutorial will provide valuable insights and techniques.

Through this tutorial, you will learn:
- The basic concepts of subfunctions.
- How to define subfunctions.
- Different ways to call subfunctions and how to pass parameters to them.
- How to leverage subfunction return values to simplify code logic.


## The Concept of Subfunctions

Subfunctions are a fundamental concept in programming, allowing for the modularization of code into smaller, reusable blocks.
In the context of DSL, subfunctions play a pivotal role in organizing code in a way that enhances readability, maintainability, and efficiency.

### What Are Subfunctions?

Subfunctions, also known as subroutines or methods in various programming paradigms, are essentially blocks of code designed to perform a specific task.
Once defined, these blocks can be invoked or called multiple times throughout a program, potentially with varying parameters, to execute the task for which they were designed.
This modular approach to programming helps in breaking down complex problems into manageable pieces.

### Benefits of Using Subfunctions

- **Code Reusability**: Once a subfunction is defined, it can be reused throughout the codebase, reducing redundancy and the potential for errors.
- **Improved Readability**: Subfunctions help to abstract away the complexity of operations, making the main code flow easier to read and understand.
- **Ease of Maintenance**: With distinct functionality encapsulated in subfunctions, updating and maintaining code becomes more straightforward. Changes made in a subfunction automatically propagate wherever the subfunction is called.
- **Simplified Debugging**: Debugging becomes easier as developers can isolate and test individual subfunctions, ensuring each part of the code performs as expected before integrating them into the larger system.

## Subfunction Language Spec

In this section, we will cover how to define and call subfunctions within Compass DSL, integrating these steps into a seamless workflow.

### Defining Subfunctions

In Compass DSL, defining a subfunction is more like Python.
Here we define an add function.
It takes two parameters, `a` and `b`, and returns their sum.

```py
@S.prim_func
def add(a: S.i32, b: S.i32) -> S.i32:
    return a + b
```

- `@S.prim_func` is a decorator meaning this is a prim func.
- `def` begins the definition of a subfunction.
- `add` is the subfunction name.
- `a: S.i32, b: S.i32` are the input parameters and their type annotation.
- `-> S.i32` is the return type. If omitted, the return type is void.
- `return` outputs a value from the subfunction. If omitted, the subfunction returns `None`.

### Calling Subfunctions

To use a subfunction, you call it by its name and pass the required arguments:

```py
@S.prim_func
def test():
    result = add(5, 3) # result: 8
```

### Parameters

There are many parameter types that can be passed:

- Scalar, like S.i32, S.u8
- Vector, like S.i32x8, S.u8x32
- Pointer, S.ptr


Note that due to design reasons, the function does not support recursive calls.

## Example

Here is a code snippet taken from bitonic sort:

```py
@S.prim_func
def cons_seq(src: S.ptr("i32", "shared")):
    step_num = log2(num) - 1
    for t in range(step_num):
        step = 1 << (t + 1)
        for i in range(num // step):
            if i % 2 == 0:
                sort_seq(src, i * step, (i + 1) * step, 1)
            else:
                sort_seq(src, i * step, (i + 1) * step, 0)


@S.prim_func
def bitonic_sort(src: S.ptr("i32", "global"), dst: S.ptr("i32", "global"), is_ascended: S.i32):
    shared_buf = S.alloc((num,), "i32", scope="shared")
    tid = S.get_local_id()
    step = num // 4

    S.dma_copy(shared_buf + tid * step, src + tid * step, step)
    S.barrier()

    cons_seq(shared_buf)
    sort_seq(shared_buf, 0, num, is_ascended)

    S.dma_copy(dst + tid * step, shared_buf + tid * step, step)
    S.barrier()
```

the generated code will be:

```c
void cons_seq(__local int* src) {
  int step_num = (log2(1024) - 1);
  for (int t = 0; t < step_num; ++t) {
    int step = (1 << (t + 1));
    for (int i = 0; i < (1024 / step); ++i) {
      int cse_var_2 = (i * step);
      int cse_var_1 = ((i + 1) * step);
      if ((i % 2) == 0) {
        sort_seq(src, cse_var_2, cse_var_1, 1);
      } else {
        sort_seq(src, cse_var_2, cse_var_1, 0);
      }
    }
  }
}

__kernel void bitonic_sort(__global int* src, __global int* dst, int is_ascended) {
  __local int shared_buf_buf[1024];
  __local int* shared_buf = shared_buf_buf;
  int tid = get_local_id(0);
  int step = 256;
  int cse_var_2 = (tid * step);
  int cse_var_1 = (step * 4);
  DMA1D((src + cse_var_2), (shared_buf + cse_var_2), cse_var_1, 1, 0);
  barrier(CLK_LOCAL_MEM_FENCE);
  cons_seq(shared_buf);
  sort_seq(shared_buf, 0, 1024, is_ascended);
  DMA1D((dst + cse_var_2), (shared_buf + cse_var_2), cse_var_1, 0, 0);
  barrier(CLK_LOCAL_MEM_FENCE);
  barrier(CLK_LOCAL_MEM_FENCE);
}
```


## Summary

Understanding and effectively utilizing subfunctions is crucial for any developer working with DSL.
By organizing code into logical blocks, developers can create more readable, maintainable, and efficient applications.
