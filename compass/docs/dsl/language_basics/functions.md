<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.-->

# Functions
## Function Definition

A function definition defines a user-defined function object, and it is an executable statement. However, when defined, it does not execute the function body. This gets executed only when the function is called.

The function definition of Compass DSL is generally the same as that of Python, including function name, decorator, parameter list, type annotation, etc. There are two things that require special attention:
- `S.prim_func` required for the function decorator.
- Type annotation required for function parameters.
- Return type annotation required if the function’s return type is not void.

Here are two simple examples to show function definition:
```py
# An example of single function definition
@S.prim_func
def func_add(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global"), n: S.int32):
    for i in range(n):
        c[i] = a[i] + b[i]


# An example of sub function that will return a value
@S.prim_func
def sub_func(inp: S.ptr("int32", "global")) -> S.i32:
    add_val = 0
    if inp[0] < 1:
        add_val += 2
    return add_val


@S.prim_func
def main_func(a: S.ptr("int32", "global"), b: S.ptr("int32", "global")):
    add_val = sub_func(a)
    va = S.vload(a)
    vb = S.vadd(va, add_val)
    S.vstore(vb, b)
```

## Function Execution

There are two ways to execute the entry function. The first is to use `BuildManager` to perform the
[build workflow](../getting_started/build_and_run_workflow.md), and run it on Compass NPU simulator
or hardware through the returned executor. The other is to treat it as a Python function and
directly run it on [PySim](../explanation/pysim.md).

Here is a simple example to show entry function execution:
```py
a, b = rand(n, dtype), rand(n, dtype)

# Call through Executor
bm = BuildManager(target="X2_1204")
ex = bm.build(func_add)
npu_out = np.zeros((n,), dtype=dtype)
ex(a, b, npu_out)

# Call directly in Python
py_out = np.zeros((n,), dtype=dtype)
func_add(a, b, py_out)
```
