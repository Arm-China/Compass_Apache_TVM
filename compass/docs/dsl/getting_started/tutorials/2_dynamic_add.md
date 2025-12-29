<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.-->

# Tutorial 2: Dynamic Add

Based on Tutorial 1 Static Add, in this tutorial, you will write a simple vector addition with dynamic kernel using Compass DSL.

## Inputs & Outputs

- Inputs:
  - Tensor(in0, shape=(n,), dtype="float16")
  - Tensor(in1, shape=(n,), dtype="float16")
- Output:
  - Tensor(out, shape=(n,), dtype="float16")

So you can write the primfunc like this:

```py
dtype = "float16"

@S.prim_func
def add_dynamic(in0: S.ptr(dtype, "global"), in1: S.ptr(dtype, "global"), out: S.ptr(dtype, "global"), n: S.i32):
    # func body
    ...
```

## Calculate Elements Number on LSRAM

Assume that the target is X2_1204, the total LSRAM size is 32 KB. Vector addition has two inputs and one output, and the output can reuse one of the input space. In this case, the data type with float16 needs 2 bytes. So the maximum number of elements on LSRAM is:

FP16_ELEMS_ON_LSRAM = 32 * 1024 // 2 // 2

The maximum number of vector with dtype float16 on LSRAM is:

FP16x16_ELEMS_ON_LSRAM = FP16_ELEMS_ON_LSRAM // 16

## Split Data for TECs

You can use S.get_local_size() to get total TEC count, and use S.get_local_id() to get the current TEC index. The number of Elements per TEC and offset of the current TEC can be calculated as follows:

```py
tec_cnt = S.get_local_size()
tid = S.get_local_id()

elems_per_tec = S.ceildiv(n, tec_cnt)
elems_cur_tec = S.clip(n - tid * elems_per_tec, min_val=0, max_val=elems_per_tec)

offset_cur_tec = tid * elems_per_tec
```

For example, assume that n is 10 and tec_cnt is 4.

The offset_cur_tec will be: 0, 3, 6, 9

The elems_cur_tec will be: 3, 3, 3, 1

## Calculation

Here is the main calculation process.

```py
for lsram_idx in range(S.ceildiv(elems_cur_tec, FP16_ELEMS_ON_LSRAM)):
    elems_cur_lsram = S.min(FP16_ELEMS_ON_LSRAM, elems_cur_tec - lsram_idx * FP16_ELEMS_ON_LSRAM)
    offset_cur_lsram = offset_cur_tec + lsram_idx * FP16_ELEMS_ON_LSRAM

    S.dma_copy(lsram_in0.as_ptr(dtype), in0 + offset_cur_lsram, elems_cur_lsram)
    S.dma_copy(lsram_in1.as_ptr(dtype), in1 + offset_cur_lsram, elems_cur_lsram)
    for vec_idx in range(S.ceildiv(elems_cur_lsram, vdtype.lanes)):
        lsram_in0[vec_idx] = S.vadd(lsram_in0[vec_idx], lsram_in1[vec_idx])
    S.dma_copy(out + offset_cur_lsram, lsram_in0.as_ptr(dtype), elems_cur_lsram)
```

1. Calculate current elements and offset on LSRAM.
2. Use dma_copy to move inputs data from DDR to LSRAM.
3. Use vector addition for current elements. The output reuses the inp0 space.
4. Use dma_copy to move output data from LSRAM to DDR.

## Complete Code
You can find the sample code in `PYTHON_PACKAGE_PATH/tvm/compass/dsl/samples/tutorial_2_dynamic_add.py`.
The placeholder `PYTHON_PACKAGE_PATH` represents the location where you install the Compass DSL
Python package, in general, it will be something like `~/.local/lib/python3.8/site-packages`.
