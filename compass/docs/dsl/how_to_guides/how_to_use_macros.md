<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.-->

# How to Use Macros

When implementing a kernel, the body of the PrimFunc may contain repeated chunks of code, or long pieces of code that make the PrimFunc harder to read. In such a situation, we would like to group some codes into other macros and make the main function body clean and readable.

```{warning}
**[PySim](../explanation/pysim.md) does not support macros.** If your DSL program has used a macro,
you cannot debug it directly in Python, only can run and debug it through the Compass simulator or
hardware, so we recommend using [subfunctions](./how_to_use_subfunctions.md) as more as possible.
```

## Why use Macros
The macros would only serve as a tool to help organize the original source, and make the main code clean and readable.

## Macro Principle
The Macro is similar to C preprocessor.

The macro would never have any actual calls generated to the PrimFunc body. Instead, they would be pasted (inlined) into TIR.

For macro call `macro_name(arg1, arg2, arg3, ...)`, the values are substituted into the body of the macro, and the body with the substituted values is then inserted at the point where the call is located.

## Macro Language Spec
- decorator: function that is used as a macro should be decorated with `@S.macro`
- parameters: all parameters can follow Python syntax, i.e. positional, keyword, etc. Type annotations are not required, but are allowed.
- macro call: `macro_name(arg1, arg2, arg3, ...)`, the same as the function call syntax.
- hygienic: this is the option to control the symbol resolution rules of macro. It is set to True (`@S.macro(hygienic=True)`) by default.
    - hygienic=True: if all symbols used in the macro's body are resolved to values from the location of the macro definition.
    - hygienic=False: the macro will have its symbols resolved to values at the time of macro use.

## Macro Symbol Resolution
In the following example, the symbol `n` is not in macro parameters, but used in the macro body.

Let’s see these two cases:

1. **Static Capture**: Symbol resolved to the value at the time of macro definition

    In this case, the **hygienic** is set to `True` by default. The symbol `n` is resolved at the time of macro definition, and its value is 8.
    ```py
    n = 8

    @S.macro
    def add_func(inp, out, idx):
        out[idx] = inp[idx] + n

    @S.prim_func
    def func(a: S.ptr("i32", "global"),
             b: S.ptr("i32", "global"),
             n: S.i32):
        if n > 0:
            add_func(a, b, 0)
        else:
            add_func(a, b, 1)
    ```
    The generated code is:
    ```c
    __kernel void func(__global int* a, __global int* b, int n) {
        if (0 < n) {
            b[0] = (a[0] + 8);
        } else {
            b[1] = (a[1] + 8);
        }
    }
    ```
2. **Dynamic Capture**: Symbol resolved to the value at the time of macro use.

    In this case, you should explicitly set `hygienic=False`. The symbol of `n` will be resolved to the value at the time of macro use.
    ```py
    n = 8

    @S.macro(hygienic=False)
    def add_func(inp, out, idx):
        out[idx] = inp[idx] + n

    @S.prim_func
    def func(a: S.ptr("i32", "global",
             b: S.ptr("i32", "global",
             n: S.i32):
        if n > 0:
            add_func(a, b, 0)
        else:
            add_func(a, b, 1)
    ```
    The generated code will be:
    ```c
    __kernel void func(__global int* a, __global int* b, int n) {
        if (0 < n) {
            b[0] = (a[0] + n);
        } else {
            b[1] = (a[1] + n);
        }
    }
    ```
## Macro Example
Here we use Macro in the example of the concat operator. The concat takes 4 inputs with shape(8) and concatenates into the output of shape(4*8).
```py

c = 8
n = 4
dtype = "int8"

@S.macro(hygienic=False)
def body(inp):
    S.dma_copy(lsram, inp, c)

@S.prim_func
def concat(
    a1: S.ptr(dtype, "global"),
    a2: S.ptr(dtype, "global"),
    a3: S.ptr(dtype, "global"),
    a4: S.ptr(dtype, "global"),
    out: S.ptr(dtype, "global"),
):
    lsram = S.alloc_buffer(dtype=dtype, shape=[c], scope="lsram")
    for tec_i in S.tec_range(0, 4):
        if tec_i == 0:
            body(a1)
        if tec_i == 1:
            body(a2)
        if tec_i == 2:
            body(a3)
        if tec_i == 3:
            body(a4)
        S.dma_copy(out + tec_i * c, lsram, c)
```
In this example, the macro body has 3 symbols:
- `inp`: in the macro parameter
- `c`: defined as const c=8
- `lsram`: not defined at the time of macro definition

If we use `@S.macro`, with hygienic set to True by default, the symbol will be resolved at the time of macro definition. However, `lsram` is not defined at the time of macro definition. Then you will get the error:
```
error: Undefined variable: lsram
```
You can solve this problem in two ways:
1. Use `@S.macro(hygienic=False)`, then the symbol of `lsram` will be resolved at the time of macro use, with lsram defined with this code:
   ```py
   lsram = S.alloc_buffer(dtype=dtype, shape=[c], scope="lsram")
   ```
   The generated code will be:
   ```c
    __kernel void concat(__global char* a1, __global char* a2, __global char* a3, __global char* a4, __global char* out) {
        int tid = get_local_id(0);
        __lsram char lsram[8];
        if (tid == 0) {
            DMA1D(a1, lsram, 8, 1, 0);
        }
        if (tid == 1) {
            DMA1D(a2, lsram, 8, 1, 0);
        }
        if (tid == 2) {
            DMA1D(a3, lsram, 8, 1, 0);
        }
        if (tid == 3) {
            DMA1D(a4, lsram, 8, 1, 0);
        }
        DMA1D((out + (tid * 8)), lsram, 8, 0, 0);
    }
   ```
2. Put `lsram` into macro parameters and use the default `@S.macro(hygienic=True)`.
   ```py
    c = 8
    n = 4
    dtype = "int8"

    @S.macro
    def body(inp, lsram):
        S.dma_copy(lsram, inp, c)

    @S.prim_func
    def concat(
        a1: S.ptr(dtype, "global"),
        a2: S.ptr(dtype, "global"),
        a3: S.ptr(dtype, "global"),
        a4: S.ptr(dtype, "global"),
        out: S.ptr(dtype, "global"),
    ):
        lsram = S.alloc_buffer(dtype=dtype, shape=[c], scope="lsram")
        for tec_i in S.tec_range(0, 4):
            if tec_i == 0:
                body(a1, lsram)
            if tec_i == 1:
                body(a2, lsram)
            if tec_i == 2:
                body(a3, lsram)
            if tec_i == 3:
                body(a4, lsram)
            S.dma_copy(out+ tec_i * c, lsram, c)
   ```
   The generated code is:
   ```c
    __kernel void concat(__global char* a1, __global char* a2, __global char* a3, __global char* a4, __global char* out) {
    int tid = get_local_id(0);
    __lsram char lsram[8];
    if (tid == 0) {
        DMA1D(a1, lsram, 8, 1, 0);
    }
    if (tid == 1) {
        DMA1D(a2, lsram, 8, 1, 0);
    }
    if (tid == 2) {
        DMA1D(a3, lsram, 8, 1, 0);
    }
    if (tid == 3) {
        DMA1D(a4, lsram, 8, 1, 0);
    }
    DMA1D((out + (tid * 8)), lsram, 8, 0, 0);
    }

   ```
   **In summary**, if you put all the symbols used in the macro body in macro parameters, you can just use `@S.macro`. If you want to only put the variant one in macro parameters, with other common symbols defined at the time of macro use, use `@S.macro(hygienic=False)`.
