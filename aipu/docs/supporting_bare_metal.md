<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.-->

# Supporting Bare Metal

Befor you read this doc, ensure you have read the content of chapter 4 of Zhouyi_Compass_Apache_TVM_User_Guide.
Bare metal environment requires hardware corresponding BSP packages. This doc introduces the method how to
run the compiled model on uboot without the bsp package. To run on uboot, you need the following parts:
- model library format(MLF): The tar ball exported through the microTVM interface contains the files of the
  compiled NN modeland the standard C runtime of the TVM,
  which can run on any device that supports standard C.
- NPU driver: The library specially developed for bare metal devices, usually named libaipu.a
- NPU driver wrapper: A simple encapsulation of the interface provided by the NPU driver, mainly called by
 the generated model C files of MLF.
- The function which calling MLF and NPU driver wrapper, and it will be called by uboot inner interface.
- Uboot source project

## Compiling and exporting an NN model

See 4.2 section of Zhouyi_Compass_Apache_TVM_User_Guide.

## Uboot interface

After compiling and exporting the NN model, you can get a tar ball containing the
model implementation and TVM standard C runtime. The format of this tar package is
completely in accordance with the standard model library format of TVM. This tar ball
is almost same as 4.3.1 section of Zhouyi_Compass_Apache_TVM_User_Guide, but the structure
of the codegen folder include an addition  file named “aipu.bin”, see as follow:

```
`-- host
|-- include
|   `-- tvmgen_default.h
`-- src
    |-- aipu.bin
    |-- default_lib0.c
    |-- default_lib1.c
    `-- default_lib2.c
```

The header file is the calling interface provided to the outside, where the specific interface is.

```c
int32_t tvmgen_default_run(
struct tvmgen_default_inputs* inputs,
struct tvmgen_default_outputs* outputs
);
```

Therefore, you only need to call this interface in one function which will integration into uboot
cmd system to run the model on the NPU.Note that, at least two parameters are required for the
function.One parameter is a void type pointer that point to the aipu.bin address, another
parameter is a void type pointer too and point to the input.bin address. Of course you can provide
more parameters if you want to do more things.

## Compiling static library of uboot cmd interface
Unzip the MLF tar ball and put it in the same directory as the files mentioned in 4.3.1 MLF tar ball,
and compile it into 4 static library file (.a file).Arm China provides a template CMake file for your
reference. The template CMake file needs to set the AIPU_TVM_BM_DEVICE_COMPILER_X2 environment variable
to specify the compiler to be used, and the ZHOUYI_BM_DRIVER_HOME environment variable to specify the
NPU bare metal driver root directory. After you finish cmake and make action, you will get 4 static
library files named libtvm_runtime.a, libcodegen.a, libaipu_driver_wrapper.a, libaipu_run_uboot.a.
This 4 static library will link into uboot for extending the cmd of uboot.

## Extend uboot cmd system
To call the interface of running model on NPU，you need to add a Custom commands of uboot.
Follow code is a sample.

```c
#include <dm/uclass-internal.h>
#include <memalign.h>
#include <asm/byteorder.h>
#include <asm/unaligned.h>
#include <asm/io.h>
#include <part.h>
#include <aipu_api.h>
#include <aipu_run.h>
#include <aipu_arch.h>

static int aipu_run_tvm(cmd_tbl_t *cmdtp, int flag, int argc, char * const argv[])
{
    unsigned long gbin = 0;
    unsigned long intput_ = 0;
    unsigned long output = 0;
    gbin = simple_strtoul(argv[1], NULL, 16);
    intput_ = simple_strtoul(argv[2], NULL, 16);
        output = simple_strtoul(argv[3], NULL, 16);
    run_aipu_on_uboot(gbin, intput_, output);
    return 0;
}

U_BOOT_CMD(
    aipu_run_tvm, 4,  1,  aipu_run_tvm,
    "run aipu tvm on uboot",
    "loadAddr dev:part"
);
```

Now, compile the uboot you can get a addition command named “aipu_run_tvm”.
you can call thiscommand with the right args to run the nn model compile by tvm.
Note that, when you compile uboot with the extend cmd code, you need link the 4 static library.

## Example
Arm China provides an example of bare metal based on juno. You can check the following files:
- out-of-box-test/tf_mobilenet_v1_bare_metal_X2/CMakeLists.txt for how to compile each part.
- out-of-box-test/tf_mobilenet_v1_bare_metal_X2/tf_mobilenet_v1.py for how the whole flow works.



