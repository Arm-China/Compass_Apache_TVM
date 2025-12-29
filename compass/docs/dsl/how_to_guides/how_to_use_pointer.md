<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.-->

# How to Use a Pointer
This section describes how to use a pointer in Compass DSL.

Here is the outline:
- Define a pointer
- Use S.ptr(dtype) as the function argument
- Match ptr with the specific shape
- Convert Scalar Ptr to Vector ptr
- Use ptr with offset
- Use ptr and buffer in vload, dma_copy


## Define a Pointer
In Compass DSL, we define a pointer with dtype and scope.
```py
a = S.ptr(dtype,scope="global")
```
- dtype:
  - Scalar dtype:`"int8","uint8","int16","uint6","int32","uint32","float16","float32","void"`
  - Vector dtype:`"int8x32","uint8x32","int16x16","uint16x16","int32x8","uint32x8","float16x16","float32x8"`

  You can also see dtype information in [Data Type](../language_basics/types.md).

  **Note**: If you want to use the void pointer, set `dtype="void"`.
- scope:

  - **global:** Represents the global DDR space of Address Space Extension region ID (ASID) 0.
  - **global.1:** Represents the global DDR space of ASID 1.
  - **global.2:** Represents the global DDR space of ASID 2.
  - **global.3:** Represents the global DDR space of ASID 3.
  - **private:** Represents the stack space of each TEC.
  - **lsram:** Represents the local SRAM space of each TEC.
  - **shared:** Represents the shared SRAM space between all TECs in the same core.
  - **constant:** Represents the global constant DDR space.

  For more information about memory hierarchy, see [Zhouyi NPU Architecture](../explanation/zhouyi_npu_arch.md).

## Use Pointer in Function Argument
You can use `S.ptr(dtype)`, with specific dtype, but without specific shape at the function argument. For 1-Dimension data, you can directly access the data element with index:
```py
@S.prim_func
def func(a: S.ptr("int8", "global"), b: S.ptr("int8", "global"), n: S.int32):
    for i in range(n):
        b[i] = a[i]
```
The generated code:
```c
__kernel void func(__global char* a, __global char* b, int n) {
  for (int i = 0; i < n; ++i) {
    b[i] = a[i];
  }
}
```

## Match Pointer with Specific Shape
For multi-dimension data, you can use `S.match_buffer` to match the ptr with the specific shape.
```py
@S.prim_func
def func(A: S.ptr("int8", "global"), B: S.ptr("int8", "global"), h: S.int32, w: S.int32):
    a = S.match_buffer(A, shape=(h, w))
    b = S.match_buffer(B, shape=(w, h))

    for i, j in S.grid(h,w):
        b[j, i]=a[i, j]
```

In the above example, `A` is a `tir.Pointer`, and `a` is a `tir.Buffer`, which supports multi-dimension index access `a[i, j]`.

The generated code:
```c
__kernel void func(__global char* a, __global char* b, int h, int w) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      b[((j * h) + i)] = a[((i * w) + j)];
    }
  }
}
```

## Convert Scalar Pointer to Vector Pointer

Here is the first example with scalar dtype
```py
@S.prim_func
def func(a: S.ptr("int8", "global"), b: S.ptr("int8", "global"), n: S.int32):
    for i in range(n):
        b[i] = a[i]
```
The generated code:
```c
__kernel void func(__global char* a, __global char* b, int n) {
  for (int i = 0; i < n; ++i) {
    b[i] = a[i];
  }
}
```

Here is the example with vector dtype. Note that the tail case is with scalar dtype.
```py
@S.prim_func
def func(a: S.ptr("int8", "global"), b: S.ptr("int8", "global"), n: S.int32):
    va = a.as_ptr("int8x32")
    vb = b.as_ptr("int8x32")

    for i in range(n // 32):
        vb[i] = va[i]

    tail_offset = n // 32 * 32
    for i in range(n % 32):
        b[tail_offset + i] = a[tail_offset + i]
```
**Note**: If you want to use the vector dtype, you should deal with the non-divisible cases manually.

## Use Pointer with Offset
A simple example shows how to create a ptr with offset.
```py
@S.prim_func
def func(a: S.ptr("int8", "global"), b: S.ptr("int8", "global")):
    a1 = a + 8       # create a new ptr with offset
    for i in range(8):
        b[i] = a1[i]
```
The generated code:
```c
__kernel void func(__global char* a, __global char* b) {
  __global char* a1 = (a + 8);
  for (int i = 0; i < 8; ++i) {
    b[i] = a1[i];
  }
}
```

This function is the sample implementation of `b[0:8] = a[8:16]`.

## Use Pointer and Buffer in vload, dma_copy

For `S.vload, S.vstore, S.dma_copy`, they support both types `tir.Pointer/tir.Buffer` for addr arguments.

- If offset=0, directly use it:
  ```py
  @S.prim_func
  def func(a: S.ptr(dtype, "global")):
    lsram = S.alloc_buffer([512], dtype, scope="lsram")
    va = S.vload(a) # a: ptr
    vx = S.vload(lsram) # lsram: buffer

    S.dma_copy(lsram,a,8)
  ```
- If offset is not zero, use `buffer.addr_of(offset)` or `ptr + offset`:
  ```py
  @S.prim_func
  def func(a: S.ptr(dtype, "global")):
      # buffer
      lsram = S.alloc_buffer([32], dtype, scope="lsram")
      # ptr
      lsram_ptr = lsram.addr_of(0)

      offset = 8

      # lsram.addr_of(offset)
      S.dma_copy(lsram.addr_of(offset), b + offset, 8)
      va = S.vload(lsram.addr_of(offset))
      # lsram_ptr + offset
      S.dma_copy(lsram_ptr + offset, b + offset, 8)
      va = S.vload(lsram_ptr + offset)
  ```
