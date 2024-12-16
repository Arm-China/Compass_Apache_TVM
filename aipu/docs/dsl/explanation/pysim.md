<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.-->

# Python Simulator (PySim)

## What is PySim?
PySim is Python Simulator, which is a very important feature of Compass DSL. In Compass DSL, the kernel
function decorated with `S.prim_func` is callable, and each API used in the kernel function has a
corresponding implementation in Python, so we can run and debug it in Python.

## Why PySim?
During the writing process of operator programs, most bugs encountered are the simple logical
errors. However, during the debugging process, each modification needs to be recompiled and deployed
to the Compass NPU Simulator or real device to run. Such repeated operations back and forth are very
time-consuming, and after the program is compiled, you need to use the Compass OpenCL Debugger to
debug the OpenCL code, which also brings greater difficulties.

Therefore, we introduced PySim to the Compass DSL. When encountering logic errors in the program,
debugging can be completed on the Python side without compiling and deploying every time, which
greatly reduces the difficulty of debugging.

## Implementation Principle of PySim
The use of Pysim is very simple. After writing a function using Compass DSL, first call BuildManager
to perform syntax compilation check, and then call the function directly in Python.
```py
import numpy as np
from tvm import aipu
from tvm.aipu import script as S

dtype = "int32"
n = 8

@S.prim_func
def func_add(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global")):
    va = S.vload(a)
    vb = S.vadd(va, 1)
    S.vstore(vb, b)

bm = aipu.tir.BuildManager()
ex = bm.build(func_add)

a = np.array(list(range(n)), dtype=dtype)
aipu_out = np.zeros((n,), dtype=dtype)
func_add(a, aipu_out)  # Run in Python(Pysim)
```

Then we use the above code as an example to describe the implementation principle of PySim.

### One Interface, Two Implementations
In order to allow an Compass DSL program to be compiled and deployed through **BuildManager**, and
use the same code to run and debug through **PySim**, we have made two implementations of each
interface, taking `S.vload` as an example:
```py
@register_ir_api
def vload(addr, mask=None, lanes=None, stride=None):
	...

@register_ir_api
def _py_vload(addr, mask=None, lanes=None, stride=None):
	...
```
For a `vload` interface, with the above two implementations, the first one is used when the Zhouyi Compass
DSL program is compiling, and the second one is used when the Zhouyi Compass DSL program is running
directly by the Python interpreter. The decorator `register_ir_api` is added to both
implementations.

Similarly, all interfaces have two implementations, and both have the decorator `register_ir_api`
added. In the module import phase of the DSL program, all interfaces will be imported into the
current namespace as module members. When importing each interface, the decorator `register_ir_api`
is executed first.
```py
def register_ir_api(func):
	...
    if func.__module__.startswith("tvm.aipu.script."):
        name = func.__name__
    else:
        # Some IR API functions of TVM Script are created through generator, so
        # their function name is same, e.g., that of T.int32 and T.uint32 is
        # "tvm.script.xxx.func_gen.<locals>.func".
        caller_code = inspect.getframeinfo(inspect.currentframe().f_back).code_context[0]
        name = caller_code.split("=")[0].strip()

    if name[0] == "_":
        assert name[:4] == "_py_", "Decorate wrong function."
        name = name[4:]
        assert _IR_API_NAME_2_IMPLEMENTS[name][1] is None, f"{name} Override happened."
        _IR_API_NAME_2_IMPLEMENTS[name][1] = func
        return func

    assert _IR_API_NAME_2_IMPLEMENTS[name][0] is None, f"{name} Override happened."
    _IR_API_NAME_2_IMPLEMENTS[name][0] = func
    ...
```
In `register_ir_api`, determine which implementation it is by the function name, and then put the
interface and its corresponding two implementation relationships into the table
`_IR_API_NAME_2_IMPLEMENTS`. Finally, a decorator function `_wrapper` is returned to allocate the
implementation when the interface is called.

After completing the two implementations of all interfaces, the next thing we have to do is to
ensure that the two execution paths of BuildManager and Pysim can get the correct implementation for
use when executing the Compass DSL program.

### Wrap Kernel Function with PyPrimFunc
In the Compass DSL program, each kernel function has a decorator `S.prim_func`.
```py
@S.prim_func
def func_add(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global")):
    va = S.vload(a)
    vb = S.vadd(va, 1)
    S.vstore(vb, b)
```
```py
@functools.wraps(T.prim_func)
def prim_func(func=None, private=False, is_entry=False):  # pylint: disable=unused-argument
    """Simple wrapper of the corresponding API of TVM Script."""

    def _decorator(myf):
        return functools.wraps(myf)(PyPrimFunc(myf, {"private": private}))

    setattr(_decorator, "dispatch_token", "tir")
    return _decorator(func) if func else _decorator
```
When calling kernel function `func_add`, you will first enter the decorator `S.prim_func`. You can
see that it returns a class instance `PyPrimFunc(myf, {"private": private})`, and `func_add` is
assigned as a parameter to the `py_func` attribute of the `PyPrimFunc`.
```py
class PyPrimFunc:
    """The simple wrapper of the Python function written by user."""

    def __init__(self, py_func, prim_func_kwargs):
        self.py_func = py_func
        self.prim_func_kwargs = prim_func_kwargs
        self.prim_func = None
        self._param_anns = []
    ...
    def __call__(self, *args):
    	...
```
At this time, the kernel function `func_add` we got is actually a `PyPrimFunc` object returned from
`S.prim_func`.

### Path to BuildManger
When we pass this kernel function to `BuildManeger`, `BuildManager` will take out `py_func` in
`PyPrimFunc` and convert it into `prim_func` in the `lower` stage, and then perform a regular [build workflow](../getting_started/build_and_run_workflow.md).

```py
def parse_to_prim_func(func):
    """Parse to TensorIR PrimFunc through TVM Script parser."""
    if not isinstance(func, PyPrimFunc):
        raise RuntimeError(
            f'The function "{func.__module__}.{func.__name__}" must be decorated by "S.prim_func".'
        )
    ret = T.prim_func(func.py_func, **(func.prim_func_kwargs))
    func.prim_func = ret
    return ret
```

### Path to PySim
When we call the kernel function directly to run on PySim, we will enter the `__call__` method of
`PyPrimFunc`.
```py
def __call__(self, *args):

    ...
    with PySimInfo() as py_sim_info:
        ...
        self.py_func(*[x.copy() if isinstance(x, PyVar) else x for x in args])
    ...
```
Here `self.py_func` will be executed. In this example, it is the kernel function `func_add`.
```py
S.prim_func
def func_add(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global")):
    va = S.vload(a)
    vb = S.vadd(va, 1)
    S.vstore(vb, b)
```
When running the kernel function `func_add`, it is wrapped with the with statement and the class
`PysimInfo` is used as the context manager.

```py
class PySimInfo:
    """Maintain all of the status information when simulate the TVM script in Python."""

    current = None

    def __init__(self, is_multi_thread=True):
        self.local_size = 4
        self.thread_local_data = threading.local()
        self.barrier = threading.Barrier(self.local_size)
        self.cur_shared_buffer = None
        self.is_multi_thread = is_multi_thread
        self._old_current = None

    def __enter__(self):
        self._old_current, PySimInfo.current = PySimInfo.current, self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        PySimInfo.current = self._old_current
```
When executing each interface after entering the `func_add` function, since `func_add` is in the
context of `PySimInfo`, as mentioned earlier, the interface returns a `_wrapper` after being
processed by `register_ir_api`. In `_wrapper`, it will be detected that the current context is under
`PysimInfo`, and then the Python version implementation of the interface will be taken out from
`_IR_API_NAME_2_IMPLEMENTS` to run and return the result.

```py
@functools.wraps(func)
def _wrapper(*args, **kwargs):
    if PySimInfo.current is None:
        return _IR_API_NAME_2_IMPLEMENTS[name][0](*args, **kwargs)

    # Execute the PySim version of the IR API.
    tld = PySimInfo.current.thread_local_data
    old_value, tld.is_in_ir_api = tld.is_in_ir_api, True
    ret = _IR_API_NAME_2_IMPLEMENTS[name][1](*args, **kwargs)
    tld.is_in_ir_api = old_value
    return ret

return _wrapper
```

### Multi-Thread to Simulate All Circumstances
Since the Zhouyi NPU architecture is multi-TEC parallel, PySim also supports multi-thread
parallelism, and each thread simulates the execution of a TEC.

```py
def __call__(self, *args):
	...
    with PySimInfo(is_multi_thread=True) as py_sim_info:

        def _run(future, thread_id):
            py_sim_info.thread_local_data.id = thread_id
            py_sim_info.thread_local_data.is_in_ir_api = False

            try:
                self.py_func(*[x.copy() if isinstance(x, PyVar) else x for x in args])
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(None)

        futures = []
        for i in range(py_sim_info.local_size):
            future = concurrent.futures.Future()
            threading.Thread(target=_run, name=f"TEC{i}", args=(future, i)).start()
            futures.append(future)

        for future in futures:
            # The exceptions raised in the sub-thread will be re-raised
            # here, so the main thread can catch and handle them.
            future.result()
	...
```

### Single-Thread Loop to Simulate Simple Circumstances (Needn’t Sync)
Of course, to facilitate debugging, we also support single-threaded operation. You only need to set
an environment variable `export AIPU_TVM_PYSIM_SINGLE_THREAD=TRUE`. After all, multi-threaded
debugging is still a relatively complicated matter. If you find that it can be reproduced with a
single thread, then you can debug the program in a single thread.

```py
if os.getenv("AIPU_TVM_PYSIM_SINGLE_THREAD", "true").upper() == "TRUE":
    WARN("PySim is running in single thread, some data race issues may can't be caught.")
    with PySimInfo(is_multi_thread=False) as py_sim_info:
        for i in range(py_sim_info.local_size):
            py_sim_info.thread_local_data.id = i
            py_sim_info.thread_local_data.is_in_ir_api = False

            self.py_func(*[x.copy() if isinstance(x, PyVar) else x for x in args])
```
