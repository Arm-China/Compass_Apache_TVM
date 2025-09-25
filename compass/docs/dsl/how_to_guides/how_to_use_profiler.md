<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.-->

# How to Use Profiler

In Compass DSL, program performance is a very important indicator. In the process of program writing, we need to constantly optimize the performance of the program. When optimizing, in addition to knowing the performance of the entire program, when analyzing performance bottlenecks, you also need to know the number of cycles of a certain piece of code.

Therefore, we provide the profile interface to obtain the performance data of the program. This interface can not only monitor the performance of the entire program, but also obtain performance data of specified code fragments by instrumenting the program.

This interface uses the RPC function (for the RPC part, please refer to [How to use RPC](./how_to_use_rpc.md)) to run the program on a real device, and automatically generates a performance data report at the end.

**Table of Contents**:
1. Insert Stubs into DSL Programs
2. Setup RPC Environment
3. Run Profile Interface
4. View Performance Reports

This article will use the Script sample program in [tutorial_0_quick_start.py](../getting_started/tutorials/0_quick_start.md) to illustrate how to use the profile interface to obtain performance data of the entire program and some of its fragments.

## 1. Insert Stubs into DSL Programs
Here is the Script sample code:
```py
@S.prim_func
def vector_add(a: S.ptr(dtype, "global"), b: S.ptr(dtype, "global"), c: S.ptr(dtype, "global"), n: S.i32):
    tec_cnt = S.get_local_size()
    tid = S.get_local_id()

    elems_per_tec = S.ceildiv(n, tec_cnt)
    cur_tec_offset = tid * elems_per_tec
    cur_tec_elems = S.clip(n - cur_tec_offset, min_val=0, max_val=elems_per_tec)

    cur_idx = cur_tec_offset
    S.perf_tick_begin(0)
    for _ in range(cur_tec_elems // 8):
        c[cur_idx : cur_idx + 8] = a[cur_idx : cur_idx + 8] + b[cur_idx : cur_idx + 8]
        cur_idx += 8
    S.perf_tick_end(0)

    remain_elems = cur_tec_elems % 8
    if remain_elems != 0:
        mask = S.tail_mask(remain_elems, 8)
        va = S.vload(a + cur_idx, mask=mask)
        vb = S.vload(b + cur_idx, mask=mask)
        vc = S.vadd(va, vb, mask=mask)
        S.vstore(vc, c + cur_idx, mask=mask)
```
We added `S.perf_tick_begin(0)` and `S.perf_tick_end(0)` before and after the for loop in the above code, which represent the start and end of tick counting respectively. In other words, we want to monitor the performence of this for loop code. The argument `0` here represents the custom ID, which can be understood as a unique identifier of this for loop code. When there are multiple instrumentations in a program, the custom ID of the code fragment corresponding to each instrumentation must be different, so that in the performance data report generated after the program is run, the data can be matched to different code snippets based on different custom IDs.

## 2. Setup RPC Environment
For the construction of the RPC system, please refer to [How to use RPC](./how_to_use_rpc.md), and then set the following environment variables.
```shell
export CPS_TVM_RPC_TRACKER_IP="192.168.1.0"
export CPS_TVM_RPC_TRACKER_PORT="9190"
export CPS_TVM_RPC_KEY="your_key"
export CPS_TVM_DEVICE_COMPILER="/xxx/aarch64-linux-gnu-g++"
```

## 3. Run Profile Interface
Build the DSL program and call the Executor's profile interface.
```py
...
print("1. Build...")
bm = BuildManager()
ex = bm.build(vector_add)
print("===============   PASS !  ===============\n")
...
print("Switch to execute on hardware device through RPC\n")
# Change parameter "rpc_key" to use your expected hardware device,
# "None" means get it from the environment variable "CPS_TVM_RPC_KEY".
ex.rpc_sess = get_rpc_session(session_timeout=60, rpc_key=None)
...
print("6. Performence test through Compass profiler...")
ex.profile(a, b, npu_out, n)
print("===============   PASS !  ===============\n")
```
After the profile interface finishes running, the following information will be output:
```shell
6. Performence test through Compass profiler...
[14:44:31][INFO] Downloaded "runtime/profile_data.bin" into "compass_dsl_vector_add_02c6cd30b6c64260a321a756fc35ab4c".
Total cycles from profiler: 3807
For more details about the profiler report, please see "compass_dsl_vector_add_02c6cd30b6c64260a321a756fc35ab4c/runtime/profile_output.html"
===============   PASS !  ===============
```
You can see that "Total cycles" is 3807, which means that the number of cycles of the entire program is 3807. The performance of the code snippets that we instrumented and monitored can be obtained by viewing "profile_output.html".

## 4. View Performance Reports
The profile_output.html is located in the runtime directory under the output folder whose name is prefixed with `compass_dsl`, in this case, the file path of profile_output.html is `compass_dsl_vector_add_02c6cd30b6c64260a321a756fc35ab4c/runtime/profile_output.html`.

Then you can use Compass_IDE to get the performance infomation. For more information, please refer to the Zhouyi CompassStudio relevant documents.
