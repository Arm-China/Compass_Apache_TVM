<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.-->

# How to use AIFF
This section describes how to use AIFF in Compass DSL.

AIFF is a specific accelerator unit inside the Zhouyi NPU with powerful computing capabilities to perform a lot of calculations.

This tutorial does not provide details of how to configure AIFF, but focuses on how to use AIFF in Compass DSL.


## Overview

In general, the way you write kernel functions has not changed much. The main change is in unittest.

There is a core concept in AIFF called descriptor. A descriptor is actually an array of register values, and in the kernel function we treat it like a pointer.

In reality, descriptors are usually generated in plugins. However, when using DSL in unittest, the plugins are not actually called, so no descriptor is generated. Therefore, we introduced the AIFF class to easily generate descriptors in unittest.

In order to use AIFF in DSL, the main work is to correctly configure the AIFF class in unittest and generate the corresponding descriptor.

## Write a kernel

### Function parameter

In a function, the descriptor is considered to be a pointer from outside.

```py
def aiff_func(
    inp: S.ptr("u8", "global"),
    out: S.ptr("u8", "global"),
    desc: S.ptr("u32", "global")
):

```

Here we receive a descriptor parameter, whose type is a pointer to uint32.
Also, since there are three different descriptor types, you can write as follows:

```py
@S.prim_func
def aiff_func(
    inp: S.ptr("u8", "global"),
    out: S.ptr("u8", "global"),
    ctrl: S.ptr("u32", "global"),
    param: S.ptr("u32", "global"),
    act: S.ptr("u32", "global")
):
```

Note that inputs and outputs can be omitted if not needed.
That is, the code can be simplified to:

```py
@S.prim_func
def aiff_func(
    ctrl: S.ptr("u32", "global"),
    param: S.ptr("u32", "global"),
    act: S.ptr("u32", "global")
):
```


### Function body
Inside the function body, you can start AIFF by the descriptor parameter.
We provide two interfaces:

- S.aiff(ctrl_desc, param_desc, act_desc)
- S.async_aiff(ctrl_desc, param_desc, act_desc, event)


Since these descriptors are treated as pointers, there are various different ways to use them inside the function.

```py
S.aiff(ctrl, param, act)
S.aiff(desc + 72 + 48, desc, desc + 72)
S.async_aiff(ctrl + 464, param + 88, act + 72, ev1)
```

In addition, any DSL features are supported.
For example, flexible width vector, Python interact and RPC.

```py
@S.prim_func
def aiff_func(out: S.ptr("u8", "global"), desc: S.ptr("u32", "global")):
    if S.get_local_id() != 0:
        return

    S.aiff(desc + 72 + 48, desc, desc + 72)

    for i in range(75 * 75 * 32):
        out[i] += 1
```

## Configure AIFF

The AIFF class is a user-interactive class that facilitates configuration of the AIFF registers.

### Create an AIFF object

There are three ways to create an AIFF object:

#### Empty

You can create an empty AIFF object directly.

```py
aipu.tir.Aiff()
```

In this case you need to configure all AIFF registers.

#### From a File

You can create an AIFF object from a descriptor file.

The descriptor file can be produced by other Zhouyi Compass utils like `aipurun`, whose name usually is `temp.desc`.

```py
aipu.tir.Aiff(descriptor=file_path)
```

In this case, only the control part of the descriptor will be used, because the other part is meaningless.

#### From an Array

You can create an AIFF object from an Array.

```py
ctrl = np.array([0x0] * 104, dtype="uint32")
ctrl[[0, 3, 8, 16, 17, 24, 32, 33, 34, 42, 43, 44, 45, 48, 49, 50]] = (0x81, 0xD, 0x20000, 0x10170, 0x3, 0x70220, 0x1E0620, 0x2, 0x2212, 0x960096, 0x960096, 0x4B004B, 0x1, 0x200020, 0x4B0096, 0x4B0096)
ctrl[[57, 58, 60, 64, 65, 80, 83, 84, 87, 96]] = (0x12C0, 0xAFC80, 0xAFC80, 0x90040, 0x22100002, 0x90050, 0x960, 0x2BF20, 0x2BF20, 0x40020)

aiff = aipu.tir.Aiff(descriptor=ctrl)
```

In this case, the array values are from the dumped descriptor file.


### Configure AIFF registers

In some cases (such as descriptor chain), a single AIFF register configuration is not enough, so there can be multiple register configurations in an AIFF class.

Note that each register configuration represents once AIFF execution. In other words, each register configuration will generate one node in the descriptor chain.

Each AIFF object represents once start interaction between the TEC and AIFF. After the start interaction, the AIFF can execute multiple times autonomously. As for how many times it will be run, it depends on the count of register configurations in the object. In other words, each object will generate one descriptor chain.

#### Add new register config (Optional)

We provide an interface to add a new AIFF register configuration.
- add_new_register_config(self, idx=None, copy_idx=None)

The idx parameter means where the new register configuration is inserted. The default is the last one.

The copy_idx parameter means which previous register configuration to copy. Because in most cases only a few places in the new register configuration need to be changed, it is more efficient to copy the previous configuration.

Use the interface as follows:

```py
aiff.add_new_register_config(copy_idx=0)
```

#### Configure value registers

The configuration format is aiff.func_unit.register.field_info.

Func unit:
- mtp
- ptp
- itp
- wrb
- unb

Register:
- act_c_ctrl
- wt_compress
- ...

Field info:
- iact_chan
- oact_chan
- ...

There may be multiple func units. You can use an index to specify a specific unit.

```py
aiff.mtp[0].act_c_ctrl.iact_chan = 1056
aiff.mtp[0].act_c_ctrl.oact_chan = 96
aiff.mtp[0].wt_compress.compression_format = 6
aiff.mtp[0].wt_compress.wt_size = 2598
aiff.mtp[1].act_c_ctrl.iact_chan = 1056
aiff.mtp[1].act_c_ctrl.oact_chan = 80
aiff.mtp[1].wt_compress.compression_format = 6
aiff.mtp[1].wt_compress.wt_size = 2550
aiff.wrb.mode_ctrl.region_en = 3
```

Here `mtp` means mtp of the last register configuration.
If you want to change the previous configuration, you can specify the index of the register configuration:

```py
aiff.reg_cfgs[0].mtp[0].act_c_ctrl.iact_chan = 1056
aiff.reg_cfgs[0].mtp[0].act_c_ctrl.oact_chan = 96
```

#### Configure address registers

There are some registers that need to be configured with Tensor addresses. Since there is no concept of address in Python, you can configure the tensor directly, and it will be automatically replaced with the address of the tensor. The tensor that needs to be configured in the register is created in unittest.


```py
inp = np.random.rand(1, 256, 256, 1024)
out = np.empty((1, 256, 256, 1024))

aiff.ptp.iact_addr = inp
aiff.wrb.region1_oact_addr = out
```

In addition, sometimes you need to configure the offset of a tensor, which can be conveniently represented by NumPy slices.

```py
aiff.reg_cfgs[0].wrb.region1_oact_addr = out[:, :, :, 64:]
aiff.reg_cfgs[1].wrb.region0_oact_addr = out[:, :, :, 128:]
aiff.reg_cfgs[1].wrb.region1_oact_addr = out[:, :, :, 224:]
```

### Generate the descriptor

Finally, you only need to call the `gen_descriptor` of the AIFF class to generate the descriptor required for operation.

```py
return aiff.gen_descriptor()
```

The returned descriptor is an object of the `DescChainArray` class. It is an array which contains the descriptor chain of ctrl, param and act.

The default order of returned descriptors is ctrl, param, act. You can combine them in any way you want.

```py
desc_chain_arr = aiff.gen_descriptor()
return desc_chain_arr.param + desc_chain_arr.act + desc_chain_arr.ctrl
```

Here reorder it as param, act, ctrl.

## Run

There is no change to the way DSL runs. The only thing to note is that the parameters passed in need to correspond to the parameters declared in the kernel function.

For a single descriptor parameter:

```py
@S.prim_func
def aiff_func(
    desc: S.ptr("u32", "global")
):
    ...

def test_aiff():
    dtype = "uint8"
    inp = rand((1, 150, 150, 32), dtype)
    gt_out = get_gt(inp)
    aiff = get_aiff(inp)

    bm = aipu.tir.BuildManager()
    ex = bm.build(aiff_func)

    py_out = np.empty((1, 75, 75, 32), dtype=dtype)
    desc = get_desc(aiff, py_out)
    aiff_func(py_out, desc)
    testing.assert_allclose(py_out, gt_out)

    aipu_out = np.empty((1, 75, 75, 32), dtype=dtype)
    desc = get_desc(aiff, aipu_out)
    ex(aipu_out, desc)
    testing.assert_allclose(aipu_out, gt_out)
```

For multiple descriptor parameters:

```py
@S.prim_func
def aiff_func(
    ctrl: S.ptr("u32", "global"),
    param: S.ptr("u32", "global"),
    act: S.ptr("u32", "global")
):
    ...

def test_aiff():
    ...
    aiff_func(py_out, desc.ctrl, desc.param, desc.act)

    ...
    ex(aipu_out, desc.ctrl, desc.param, desc.act)
```

## Multi-AIFF

Multiple physical AIFFs exist simultaneously in the Zhouyi X3 target. Obviously, in this case, multiple AIFFs will start, and one AIFF object is not enough.

The solution is to create multiple AIFF objects. The key is to organize the relationship between descriptors.

```py
desc0 = aiff0.gen_descriptor()
desc1 = aiff1.gen_descriptor()

# Puts all ctrl together and so on.
def puts_together(desc0, desc1):
    return desc0.ctrl + desc1.ctrl + desc0.param + desc1.param + desc0.act + desc1.act

# Or puts every chain independently.
def puts_independently(desc0, desc1):
    return desc0 + desc1
```