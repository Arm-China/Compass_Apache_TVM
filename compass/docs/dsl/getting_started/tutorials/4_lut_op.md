<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.-->

# Tutorial 4: LUT Operator

In this tutorial, you will write a simple vector lookup-table (LUT) operator to implement silu using Compass DSL. You will learn about:
* How to call a subroutine function on the device side.
* The difference between running on the host side when compling and function running on the device side during runtime.
* How to declare a constant LUT table.
* How to vectorize using table-lookup vector-builtin to implement silu using the pre-prepared LUT table.
* How to interpolate between 2 items to implement higher precision Silu.


## Inputs & Outputs

- Inputs:
    - Tensor(in0 ,shape=[n], dtype="float16")
    - n: i32
- Output:
    - Tensor(out0 ,shape=[n], dtype="float16")

So we can write the primfunc like this:
```py
dtype = "float16"
vdtype = "float16x16"
lut_len = 512
lut_edge = 10

FP16_ELEMS_ON_LSRAM = 32 * 1024 // 2
FP16x16_ELEMS_ON_LSRAM = FP16_ELEMS_ON_LSRAM // 16

@S.prim_func
def silu_fp16(
    in0: S.ptr(dtype, "global"),
    out0: S.ptr(dtype, "global"),
    n: S.i32,
):
    # func body
    ...
```
## Prepare LUT Table
Assume that the input value ranges from [-10, 10] and we will prepare a table with 512 items.

- In this case, we assume that the input value ranges from [-10, 10].
- We prepare a table with 512 items.


```py
def gen_silu_lut(lut_len, lut_edge, dtype):
    # Key points of FP16 LUT generation:
    # 1. lut_x[-lut_edge, lut_edge]
    # 2. dytpe = "float16"
    # 3. pad the lut_table with the last element (lut_edge)
    x = np.linspace(-lut_edge, lut_edge, lut_len - 1)
    n = lut_len - 1
    lut = np.zeros((n,), dtype=dtype)
    for i in range(n):
        lut[i] = x[i] / (1 + np.exp(-x[i]))
    lut.resize(lut_len)
    lut[-1] = lut_edge
    return lut

@S.prim_func
def silu_fp16(
    in0: S.ptr(dtype, "global"),
    out0: S.ptr(dtype, "global"),
    n: S.i32,
):
    lut = S.alloc_const((lut_len,), dtype, gen_silu_lut(lut_len, lut_edge, dtype))
    ...
```
You may notice that:
- gen_silu_lut is not decorated by S.prim_func, which means that:
    - This function would be running on the host side during the compiling time.
    - This function would not generate any code on the device side.
- S.alloc_const means allocating a “__constant” buffer on OpenCL and its value is determined by the gen_silu_lut return value.


## Kernel Function

Here is the main kernel function

```py
@S.prim_func
def silu_fp16(
    in0: S.ptr(dtype, "global"),
    out0: S.ptr(dtype, "global"),
    n: S.i32,
):
    lut = S.alloc_const((lut_len,), dtype, gen_silu_lut(lut_len, lut_edge, dtype))
    lut_inverse_delta = S.fp32(lut_len - 2) / (2 * lut_edge)

    lsram_ptr = S.alloc(FP16x16_ELEMS_ON_LSRAM, vdtype, scope="lsram")
    tec_cnt = S.get_local_size()
    tid = S.get_local_id()

    elems_per_tec = S.ceildiv(n, tec_cnt)
    elems_cur_tec = S.clip(n - tid * elems_per_tec, min_val=0, max_val=elems_per_tec)

    offset_cur_tec = tid * elems_per_tec
    for lsram_idx in range(S.ceildiv(elems_cur_tec, FP16_ELEMS_ON_LSRAM)):
        elems_cur_lsram = S.min(FP16_ELEMS_ON_LSRAM, elems_cur_tec - lsram_idx * FP16_ELEMS_ON_LSRAM)
        offset_cur_lsram = offset_cur_tec + lsram_idx * FP16_ELEMS_ON_LSRAM

        S.dma_copy(lsram_ptr.as_ptr(dtype), in0 + offset_cur_lsram, elems_cur_lsram)
        for vec_idx in range(S.ceildiv(elems_cur_lsram, vdtype.lanes)):
            lsram_ptr[vec_idx] = compute(lsram_ptr[vec_idx], lut, lut_inverse_delta, lut_len, lut_edge)
        S.dma_copy(out0 + offset_cur_lsram, lsram_ptr.as_ptr(dtype), elems_cur_lsram)
```

The main calculation process is as follows:
- Calculate how many elements should be calculated on this TEC
- Use dma_copy to copy the related input data from DDR to LSRAM
- Call device prim_func to compute silu inplace on LSRAM
- Use dma_copy to move the silu result from LSRAM to DDR

The main calucation logic is put on function “compute”. This function should be generated on the device side. So this is a device function. As a device funtion that should be run on the device during runtime, it should be decorated by S.prim_func.


## Subroutine Call Device Function

The “compute” function accepts 5 parameters:
- x, the input data, the data type is “fp16x16”, on LSRAM
- lut, the lut table we pre-prepared
- lut_inverse_delta: the gap between each item on the lut table, used to calucate the index and for interpolation.
- lut_len, the table len
- lut_edge: the table value range.
The function return value data type is “fp16x16”.
Also, it is a device function that should be decorated by S.prim_func.


So the function looks like this:
```py
    @S.prim_func
    def compute(
        x: S.fp16x16,
        lut: S.ptr(dtype, "constant"),
        lut_inverse_delta: S.fp32,
        lut_len: S.i32,
        lut_edge: S.i32,
    ) -> S.fp16x16:
    ...
```

- Clip the input value to the table range [-10, 10]
- Use S.cast to convert the input value from “fp16x16” to fp32x16 “x_fp32”
- Calculate the index and round fp32x16 "x_idx"

```py
    # in compute function.
    x_clipped = S.clip(x, min_val=S.fp16(-lut_edge), max_val=S.fp16(lut_edge))
    x_fp32 = S.cast(x_clipped, "fp32")
    x_idx = (x_fp32 + lut_edge) * lut_inverse_delta
    x_idxr = S.rint(x_idx - 0.5)
    x_idx_u16 = S.cast(x_idxr, "u16")
    ...
```

- Use S.vload_gather to look up the lut table.
- For interpolation, we also look up the next items of the lut table.
- Calculate the diff and linear interpolation. Remember to use fp32x8 for interpolation.
- After interpolation, cast down to fp16x16.

```py
    lut_x_idx = S.vload_gather(lut, x_idx_u16)
    lut_x_idx_plus1 = S.vload_gather(lut, x_idx_u16 + 1)
    x_idx_diff = x_idx - x_idxr
    lut_diff = S.cast(lut_x_idx_plus1 - lut_x_idx, "fp32")
    yy = S.cast(lut_diff * x_idx_diff, dtype)
    ...
```

- To ensure that the index does not exceed the range, calculate the mask.
- Return base + diff.

```py
    mask_idx_ge_lutlen_m2 = x_idx_u16 >= S.u16x16(lut_len - 2)
    y = S.vsel(x, 0, mask_idx_ge_lutlen_m2)
    mask_xor = S.vxor(x_idx_u16 >= 0, mask_idx_ge_lutlen_m2)
    return S.vadd(yy, lut_x_idx, mask=mask_xor, r=y)
```

The whole code is like this:
```py
@S.prim_func
def compute(
    x: S.fp16x16,
    lut: S.ptr(dtype, "constant"),
    lut_inverse_delta: S.fp32,
    lut_len: S.i32,
    lut_edge: S.i32,
) -> S.fp16x16:
    # Original formula: silu(x) = x / (1 + e^(-x))
    # Here use lookup table implement with interpolation instead
    x_clipped = S.clip(x, min_val=S.fp16(-lut_edge), max_val=S.fp16(lut_edge))
    x_fp32 = S.cast(x_clipped, "fp32")
    x_idx = (x_fp32 + lut_edge) * lut_inverse_delta
    x_idxr = S.rint(x_idx - 0.5)
    x_idx_u16 = S.cast(x_idxr, "u16")
    mask_idx_ge_lutlen_m2 = x_idx_u16 >= S.u16x16(lut_len - 2)

    y = S.vsel(x, 0, mask_idx_ge_lutlen_m2)
    lut_x_idx = S.vload_gather(lut, x_idx_u16)
    lut_x_idx_plus1 = S.vload_gather(lut, x_idx_u16 + 1)
    x_idx_diff = x_idx - x_idxr
    lut_diff = S.cast(lut_x_idx_plus1 - lut_x_idx, "fp32")
    yy = S.cast(lut_diff * x_idx_diff, dtype)
    mask_xor = S.vxor(x_idx_u16 >= 0, mask_idx_ge_lutlen_m2)
    return S.vadd(yy, lut_x_idx, mask=mask_xor, r=y)
```

## Complete Code
You can find the sample code in `PYTHON_PACKAGE_PATH/tvm/compass/dsl/samples/tutorial_4_lut_op.py`.
The placeholder `PYTHON_PACKAGE_PATH` represents the location where you install the Compass DSL
Python package, in general, it will be something like `~/.local/lib/python3.8/site-packages`.
