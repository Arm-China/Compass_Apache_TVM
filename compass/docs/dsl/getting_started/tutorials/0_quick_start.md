<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.-->

# Tutorial 0: Quick Start

In this tutorial, you will learn the basic workflow of Compass DSL. You will learn about how to:
* Write Basic Compass DSL Function
* Build the Function
* Inspect OpenCL C Code
* Run in Python ([PySim](../../explanation/pysim.md))
* Run on NPU Simulator
* Run on Remote Hardware Device through RPC
* Profile

## 1. Write Basic Compass DSL Function
You can write a simple function with decorator `S.prim_func`:
- The input and output data needs to be annotated as `S.ptr(dtype, "global")`.
- The function body is the computations.

```py
from tvm.compass.dsl import BuildManager, script as S


dtype = "float32"


@S.prim_func
def func_add(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global"), n: S.i32):
    for i in range(n):
        c[i] = a[i] + b[i]
```

## 2. Build the Function
The `BuildManager` is the user interface for DSL program compilation.
To define an instance of `BuildManager`, the `target` is required. The default target is `X2_1204`.
You can then call the `build` method of `bm` to build the DSL function and get an executor.

```py
bm = BuildManager(target="X2_1204")
ex = bm.build(func_add)
```
For more information, see [build workflow of Compass DSL](../build_and_run_workflow.md).

## 3. Inspect OpenCL C Code
After building the function, you can get the generated OpenCL C code.
```py
print(ex.c_code)
```
The generated OpenCL C code will be something like below.
```c
__kernel void func_add(__global float* a, __global float* b, __global float* c, int n) {
  for (int i = 0; i < n; ++i) {
    c[i] = (a[i] + b[i]);
  }
}
```

## 4. Run in Python ([PySim](../../explanation/pysim.md))
You can run the DSL function in Python directly.

In this case, you can directly call `func_add` by passing the appropriate arguments.
```py
n = 1024
a, b = rand(n, dtype), rand(n, dtype)
py_out = np.zeros(n, dtype=dtype)

func_add(a, b, py_out, n)
```

## 5. Run on NPU Simulator
After the build step, the `ex` is an executable object. You can directly run it by passing the
appropriate arguments.
```py
npu_out = np.zeros(n, dtype=dtype)
ex(a, b, npu_out, n)
```
You can directly print the output data.
```py
print(npu_out)
```

## 6. Run on Remote Hardware Device through RPC
Compass DSL also supports running on hardware device through RPC.

You need to set the RPC relevant environments:
```shell
export CPS_TVM_RPC_TRACKER_IP="xxx"
export CPS_TVM_RPC_TRACKER_PORT="xxx"
export CPS_TVM_RPC_KEY="xxx"
export CPS_TVM_DEVICE_COMPILER="/xxx/aarch64-linux-gnu-g++"
```
Establish an RPC session with the remote hardware device and run the function.
```py
# Switch to execute on hardware device through RPC
# rpc_key = "None" means get it from env "CPS_TVM_RPC_KEY".
ex.rpc_sess = get_rpc_session(session_timeout=60, rpc_key=None)
npu_out = np.zeros(n, dtype)
ex(a, b, npu_out, n)
```

## 7. Profile
You can use the benchmark API to get the function execution time on remote hardware device.
```py
ex.benchmark(a, b, npu_out, n)

# Execution time summary:
# mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
#   0.6287       0.6287       0.6613       0.5962       0.0326
```
If the amount of data is too small, it is recommended to repeat multiple loops to reduce measurement
errors, thereby obtaining more reliable performance data.
```py
ex.benchmark(a, b, npu_out, n, repeat=10, number=10)
```

You can also use the NPU profiler to collect accurate and detailed performance information.
```py
ex.profile(a, b, npu_out, n)

# Total cycles from profiler: 2773
# For more details about the profiler report, please see "compass_dsl_xxx/runtime/profile_output.html"
```

## Complete Code
You can find the sample code in `PYTHON_PACKAGE_PATH/tvm/compass/dsl/samples/tutorial_0_quick_start.py`.
The placeholder `PYTHON_PACKAGE_PATH` represents the location where you install the Compass DSL
Python package, in general, it will be something like `~/.local/lib/python3.8/site-packages`.
