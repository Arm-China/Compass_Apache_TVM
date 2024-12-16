<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.-->

# Flexible/Multiple Width Vector

## Background

When using vector operations, fixed width vectors may generate duplicate code. For example, in order to reduce the loss of precision, the data type will be cast up before calculation. If a vector is limited by the fixed length of the register, you need to divide the original vector into multiple sub-vectors, then write the same calculation code for each sub-vector.

```
va: float16x16
out: (va + x) * y - z

va_low = cast(va, "float32", part="low")
va_high = cast(va, "float32", part="high")
va_low_add = va_low + x
va_low_add_mul = va_low_add * y
va_low_out = va_low_add_mul - z
va_high_add = va_high + x
va_high_add_mul = va_high_add * y
va_high_out = va_high_add_mul - z
out = cast([va_low_out, va_high_out], "float16")
```

To reduce the above duplication code, we define a virtual vector type that allows you to use vectors with flexible width or multiple width.

## Flexible width vector

The flexible width vector is a virtual vector with any fixed-positive-integer lanes (greater than 1). For example, in a 256-bit register, the fixed width vector type can be int32x8, float16x16, while the flexible width vector type can be int32x100, int32x11, float16x7, float16x33.

## Multiple width vector

The multiple width vector feature is similar to the flexible width vector feature. The main differences between them include:

- The multiple width vector only supports integral multiple lanes of fixed width, such as int32x16, float16x48.
- The multiple width vector supports more APIs than the flexible width vector, such as vaddh.

## How to use a flexible/multiple width vector

### Creation

There are some methods to create a flexible/multiple width vector as follows:

1. Using any fixed region in a scalar array.

   ```
   a = S.alloc((60,), "int32")
   b = S.alloc((70,), "int32")
   a[0:60]   #i32x60
   b[0:70:2] #i32x35
   ```

2. Using a fixed number of lanes in S.vload or S.vload_gather.

   ```
   va = S.vload(a, lanes=60) #i32x60
   vb = S.vload(b, lanes=33) #i32x33
   
   indices = S.vload(idx_arr, lanes=60)
   vc = S.vload_gather(a, indices)
   ```

3. Using S.vconcat.

   ```
   va = S.vload(a) #i32x8
   vb = S.vload(b) #i32x8
   vc = S.vconcat(a, b) #i32x16
   ```

4. Using S.cast.

   ```
   va: int8x32
   vb = S.cast(va, "int32")
   vb: int32x32
   ```

### Usage

After creation, this flexible/multiple width vector can be used directly in the supported API later as a normal vector.

```
a = S.alloc((60,), "int32")
b = S.alloc((60,), "int32")
c = S.alloc((60,), "int32")
a[0:60] = 1
b[0:60] = 2
b[30:60:2] = a[0:30:2]
c[0:15] = a[0:15] + b[0:15]
c[15:30] = a[15:30] - b[15:30]
c[30:45] = a[30:45] * b[30:45]
c[45:60] = a[45:60] / b[45:60]
```

```
va = S.vload(a, lanes=60)
vb = S.vload(b, lanes=60)
out = va * scale + vb
S.vstore(out, c)

indices = S.vload(idx_arr, lanes=60)
va = S.vload_gather(a, indices)
vb = S.vload_gather(b, indices)
out = S.vadd(va, vb)
S.vstore_scatter(out, c, indices)
```

APIs that support flexible/multiple width will be noted in the DSL built-in API documentation.

There is a special usage in multiple width vectors. You can restore to multiple real vectors and merge them after performing unsupported interface operations.

```
# a: int8
va = S.vload(a, lanes=64)   # int8x64
va = va * scale + 3
va0, va1 = S.vsplit(va)
va0 = S.vreplic(va0, 3) # int8x32
va1 = S.vreplic(va1, 4) # int8x32
va = S.vconcat(va0, va1)    # int8x64
va = va / scale - 4
S.vstore(va, c)
```

### Restrictions

There are also some restrictions for this feature as follows:

- Can only be used within a function. Cross functions are unsupported now.

- Cannot define an array of virtual width vector type, such as

   ```
   # invalid code
   va = S.alloc((2, 3), "int8x100")
   ```

- Cannot be used in half-number operation mode of some instructions, such as

   ```
   # invalid code
   va: int8x100
   vb = S.cast(va, "int32", part="even")
   vb = S.cast(va, "int32", part="odd")
   vb = S.cast(va, "int32", part="low")
   vb = S.cast(va, "int32", part="high")
   ```

## Example

```
@S.prim_func
    def compute(x: S.fp16x16, alpha: S.fp32) -> S.fp16x16:
        # celu(x)
        #   = x, if x >= 0
        #   = alpha * (exp(x/alpha) - 1)), if x < 0
        x_fp32 = S.cast(x, "fp32")
        x_exp = S.exp(x_fp32 / alpha) - 1
        x_out = S.vmul(alpha, x_exp, mask=(x_fp32 < 0), r=x_fp32)
        return S.cast(x_out, "fp16")
```

It will generate OpenCL code automatically as follows:

```
half16 celu_fp16_compute(half16 x, float alpha) {
  float8 x_fp32_0 = __vzipl(__vcvtue_tfp32(x), __vcvtuo_tfp32(x));
  float8 x_fp32_1 = __vziph(__vcvtue_tfp32(x), __vcvtuo_tfp32(x));
  float8 x_exp_0 = (exp(__vdiv((float8)0.0f, x_fp32_0, (float8)alpha, ALL_TRUE_w)) - (float8)1.0f);
  float8 x_exp_1 = (exp(__vdiv((float8)0.0f, x_fp32_1, (float8)alpha, ALL_TRUE_w)) - (float8)1.0f);
  float8 x_out_0 = __vmul(x_fp32_0, (float8)alpha, x_exp_0, __vqlt(x_fp32_0, (float8)0.0f, ALL_TRUE_w));
  float8 x_out_1 = __vmul(x_fp32_1, (float8)alpha, x_exp_1, __vqlt(x_fp32_1, (float8)0.0f, ALL_TRUE_w));
  return __vcvtd(x_out_0, x_out_1);
}
```
