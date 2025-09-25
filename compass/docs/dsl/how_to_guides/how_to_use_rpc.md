<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.-->

# How to Use RPC

Remote Procedure Call (RPC) is a very important and useful feature of Apache TVM. It allows to
run a compiled program on the real hardware without the need to touch the remote device. The output result
will be passed back automatically through the network.

In Compass DSL, with the support of the RPC function, you can easily run programs written using DSL on
remote devices.

## Example

In order for the program to run on the device through RPC, you need to obtain the `rpc session`
through the TVM official API:
```py
rpc_session = rpc.connect_tracker(tracker_host, tracker_port).request(rpc_key)
```
`tracker_host`, `tracker_port` and `rpc_key` are parameters that you need to set.

Of course, you can also use the function `get_rpc_session` that we provide in the util module to obtain
the `rpc session`.

It should be noted that no matter whether it is through the TVM official API or util function, the environment
variable `CPS_TVM_DEVICE_COMPILER` needs to be set for cross-compilation of the program.
```shell
export CPS_TVM_DEVICE_COMPILER="/xxx/aarch64-linux-gnu-g++"
```

Here is a simple demo to show a program running with RPC:
```py
import numpy as np
from tvm.compass.dsl import BuildManager, script as S, get_rpc_session
from tvm.compass.dsl.testing import rand, assert_allclose


@S.prim_func
def vector_add(a: S.ptr("i8x32", "global"), b: S.ptr("i8x32", "global"), c: S.ptr("i8x32", "global")):
    c[0] = S.vadd(a[0], b[0])

n = 32
a, b = rand(n, "int8"), rand(n, "int8")
gt_out = a + b

bm = BuildManager()
ex = bm.build(vector_add)

# ---------- Get RPC session though TVM Official API ----------
# ex.rpc_sess = rpc.connect_tracker(tracker_host, tracker_port).request(rpc_key)

# ---------- Get RPC session though util function "get_rpc_session" ----------
# Change parameter "rpc_key" to use your expected hardware device,
# "None" means get it from the environment variable "CPS_TVM_RPC_KEY".
# Change parameter "tracker_host" to use your hostname or IP address of RPC tracker,
# "None" means get it from the environment variable "CPS_TVM_RPC_TRACKER_IP".
# Change parameter "tracker_port" to use your port of PRC tracker,
# "None" means get it from the environment variable "CPS_TVM_RPC_TRACKER_PORT".
ex.rpc_sess = get_rpc_session(rpc_key=None, tracker_host=None, tracker_port=None)

npu_out = np.zeros(n, "int8")
ex(a, b, npu_out)
assert_allclose(npu_out, gt_out)
```

## Reference

For more detailed infomation about how to set up the RPC system, please refer to Apache TVM official
how-to guide [Setup RPC System](https://tvm.apache.org/docs/dev/how_to/setup_rpc_system.html).
