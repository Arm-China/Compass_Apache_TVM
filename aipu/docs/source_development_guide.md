<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.-->

# Source Development Guide

## Introduction
The Apache TVM is an open-source machine learning compiler framework for CPUs, GPUs, and accelerators. It aims to enable machine learning engineers to optimize and run computations efficiently on any hardware backend.

The Zhouyi Compass is integrated with the Apache TVM for Neural Network (NN) model quick support and heterogeneous execution.

The Apache TVM parser supports extensive Machine Learning (ML) frameworks, including TensorFlow, PyTorch, ONNX, Keras, TensorFlow Lite, TensorFlow 2, Caffe, MxNet, PaddlePaddle, and Darknet. Through the Zhouyi Compass integration with the Apache TVM, your NN model can be supported quickly even though the NN compiler of the Zhouyi Compass does not support the ML framework of the NN model yet.

The Apache TVM provides the solution for graph partition and heterogeneous execution, through the Zhouyi Compass integration with the Apache TVM. The NN operators supported by Zhouyi NPU are split to Zhouyi NPU sub graph automatically, and you can continue to place the remaining NN operators to other devices (for example, CPU and GPU). After building the Apache TVM, the runtime will execute the NN model heterogeneously and automatically. So, your NN model can be supported even though some of its NN operators are not supported by the Zhouyi Compass.

This source package is developed based on the Apache TVM whose commit id is *a6157a6369c184b6fa5f66654feb685e58726737* and also includes some necessary cherry picks.

## Structure
The directory structure of the source code is the same as the Apache TVM, and the code of Arm China team is added in the following directory.
```shell
python/tvm/relay/backend/contrib/aipu_compass/
|-- __init__.py             AIPU Compass backend of Relay.
|-- _ffi_api.py             FFI APIs for AIPU Compass.
|-- aipu_builder.py         Interface with package AIPUBuilder.
|-- aipu_compass.py         Simple wrap of AIPU Compass compile flow.
|-- codegen.py              AIPU Compass IR codegen of Relay.
|-- config.py               Configuration processing of AIPU Compass.
|-- deployable.py           Code to handle the work when deploying a TVM compiled NN model.
|-- engine                  Provide a mechanism to process AIPU Compass function flexibly.
|-- execution_engine.py     Engines to handle the work when executing a TVM compiled NN model.
|-- logger.py               Logger of AIPU Compass.
|-- parser.py               Simple wrap each Relay frontend API.
|-- testing                 Provide the data and functions required by testing.
|-- transform               The Relay IR namespace containing AIPU Compass specific transformations.
`-- utils.py                Common AIPU Compass utilities.

python/tvm/relay/op/contrib/
|-- aipu_compass.py         Relay IR to Compass IR mapping rules.
|-- aipu_compute_lib.py     Custom operator strategy for AIPU.
|-- aipu_custom_op.py       Custom operator support of Relay.
|-- aipu_plugin_register.py Custom operator plugin register for aipu.

aipu/
|-- cmake                   CMake build rules for AIPU.
|-- include                 Some head file for AIPU.
|-- src                     Some C++ codes.
`-- tests                   Test cases including the E2E and Unittests.
```

In addition, some of the Apache TVM code has been modified by the Arm China team, and the mark "This file has been modified by Arm China team." has been added to the header of the file.

## Install from Source
This section gives instructions on how to build and install the TVM Package from scratch, It consists of two steps:
1. First build the shared library from the C++ codes
2. Setup for the Python package

### Setting up the environment
The following lists other Zhouyi Compass packages that are required:
- NPU Linux driver

For the installation instructions of this package, please see the *Arm China AI Platform Zhouyi Compass Software Programming Guide*. After compiling the NPU Linux driver successfully, ensure that the required dynamic library `libaipudrv.so` can be found from environment variable `LD_LIBRARY_PATH`.

Also, set `ZHOUYI_LINUX_DRIVER_HOME` like below:
```shell
export ZHOUYI_LINUX_DRIVER_HOME="xxx/AI610-SDK-1012-xxxx-xxxx/Linux-driver"
```

### Clone the 3rdparty
The submodules that tvm depends on are not included in the source release pack and need to be added back after pack is decompressed.

We provide a script in the source release pack to initialize this pack as a git repository and add submodules.

Please execute the following command in the root directory of source release pack:
```shell
bash submodule_init.sh
```

### Build the Shared Library
Our goal is to build the shared libraries:
- On Linux the target library are `libtvm.so` and `libtvm_runtime.so`
- On Target device the library is `libtvm_runtime.so`

The minimal building requirements for the TVM libraries are:
- g++-5 or higher
- CMake 3.10 or higher
- LLVM 4.0 or higher
- Python3.8.X+

After installing the required dependencies, continue to build the libraries.

#### Build for Linux
Execute the following command in the root directory of the source release pack.
```shell
cp cmake/config.cmake ./
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

#### Build for Target Device (Cross compile)
If you want to deploy application to your target platform, we must link to a TVM runtime in your target platform. Therefore, we need to modify some configurations in `cmake/config.cmake` to get cross compiled `libtvm_runtime.so`:
- Set AIPU target, change `set(USE_AIPU ON)` to `set(USE_AIPU {target platform})`, e.g. set(USE_AIPU QNX)
- Set include directory of AIPU driver, change `set(AIPU_DRIVER_INCLUDE_DIR $ENV{ZHOUYI_LINUX_DRIVER_HOME}/driver/umd/include)` to `set(AIPU_DRIVER_INCLUDE_DIR {target include dir})`, e.g. set(AIPU_DRIVER_INCLUDE_DIR path/to/qnx/include)
- Set library of AIPU driver, change `set(AIPU_DRIVER_LIB $ENV{ZHOUYI_LINUX_DRIVER_HOME}/bin/sim/release/libaipudrv.so)` to `set(AIPU_DRIVER_LIB {target device lib})`, e.g. set(AIPU_DRIVER_LIB path/to/qnx/libaipudrv.a)
- Disable llvm, change `set(USE_LLVM ON)` to `set(USE_LLVM OFF)`
- Disable libbacktrace if target platform is not Linux or macOS, change `set(USE_LIBBACKTRACE AUTO)` to `set(USE_LIBBACKTRACE OFF)`
- Disable MicroTVM runtime, change `set(USE_MICRO ON)` to `set(USE_MICRO OFF)`
- Set `CMAKE_TOOLCHAIN_FILE` in command line, e.g. -DCMAKE_TOOLCHAIN_FILE=path/to/qnx.cmake
- For other optional configuration items, please see `cmake/config.cmake`

Below is an example of cross compiling TVM runtime for QNX:
```shell
mkdir cross_build
cd cross_build
cp ../cmake/config.cmake ./
sed -i "s|USE_AIPU.*)|USE_AIPU QNX)|" config.cmake
sed -i "s|AIPU_DRIVER_INCLUDE_DIR.*)|AIPU_DRIVER_INCLUDE_DIR path/to/qnx/include)|" config.cmake
sed -i "s|AIPU_DRIVER_LIB.*)|AIPU_DRIVER_LIB path/to/qnx/libaipudrv.so)|" config.cmake
sed -i "s|USE_LLVM.*)|USE_LLVM OFF)|" config.cmake
sed -i "s|USE_LIBBACKTRACE.*)|USE_LIBBACKTRACE OFF)|" config.cmake
sed -i "s|USE_MICRO.*)|USE_MICRO OFF)|" config.cmake
cmake -DCMAKE_TOOLCHAIN_FILE=path/to/qnx.cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . -j -- runtime
```
In addition to the general settings in CMake toolchain file, below code must be added in the
toolchain file of QNX, because the current DMLC-Core project lacks the support for QNX platform.

```cmake
add_compile_definitions(DMLC_LOG_STACK_TRACE=0)
add_compile_definitions(DMLC_CMAKE_LITTLE_ENDIAN=1)
```

Below is an example of cross compiling TVM runtime for Android:

You will need [JDK](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html),
[Android SDK](https://developer.android.com/studio/index.html), [Android NDK](https://developer.android.com/ndk)
to use this. Make sure the `ANDROID_HOME` variable already points to your Android SDK folder or set
it using `export ANDROID_HOME=[Path to your Android SDK, e.g., ~/Android/sdk]`.

```shell
make jvmpkg
mkdir cross_build
cd cross_build
cp ../cmake/config.cmake ./
sed -i "s|USE_AIPU.*)|USE_AIPU android)|" config.cmake
sed -i "s|AIPU_DRIVER_LIB.*)|AIPU_DRIVER_LIB path/to/android/libaipudrv.so)|" config.cmake
sed -i "s|USE_LLVM.*)|USE_LLVM OFF)|" config.cmake
sed -i "s|USE_LIBBACKTRACE.*)|USE_LIBBACKTRACE OFF)|" config.cmake
sed -i "s|USE_MICRO.*)|USE_MICRO OFF)|" config.cmake
cmake -DCMAKE_TOOLCHAIN_FILE=${ANDROID_HOME}/ndk/21.4.7075529/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j -- runtime
${ANDROID_HOME}/ndk/21.4.7075529/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android-strip libtvm_runtime.so
mv libtvm_runtime.so libtvm4j_runtime_packed.so
```

Below is an example of cross compiling TVM runtime for Juno r2 Development Platform:
```shell
mkdir cross_build
cd cross_build
cp ../cmake/config.cmake ./
sed -i "s|USE_AIPU.*)|USE_AIPU juno)|" config.cmake
sed -i "s|AIPU_DRIVER_LIB.*)|AIPU_DRIVER_LIB path/to/juno/libaipudrv.so)|" config.cmake
sed -i "s|USE_LLVM.*)|USE_LLVM OFF)|" config.cmake
sed -i "s|USE_LIBBACKTRACE.*)|USE_LIBBACKTRACE OFF)|" config.cmake
sed -i "s|USE_MICRO.*)|USE_MICRO OFF)|" config.cmake
cmake -DCMAKE_TOOLCHAIN_FILE=path/to/juno.cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j -- runtime
```

### Python Package Installation
The python package is located at `${TVM_HOME}/python`, set the environment variable `PYTHONPATH` to tell python where to find the library.

For example, assume we decompress `dlhc-xxx.tar.gz` on the directory `/path/to/dlhc-xxx` then we can set the `TVM_HOME` to `/path/to/dlhc-xxx`.

Execute the following command to install the package:
```shell
export TVM_HOME=/path/to/dlhc-xxx
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

## Enable Tests

### Tests of the Zhouyi Compass
We use `Pytest <https://docs.pytest.org/en/7.1.x/>` to drive the tests in TVM.

The tests we provided including two types:
```shell
aipu/tests/python/
|-- __init__.py
|-- e2e        End to End tests.
`-- unittest   Unittests.
```

Before running tests, we have to set some variables:
- Set the environment variable `ZHOUYI_MODEL_ZOO_HOME` to tell python where to find the model and data file.

```shell
We provide the ftp download address for these files on GitHub, please check https://github.com/Zhouyi-AIPU/Model_zoo
    export ZHOUYI_MODEL_ZOO_HOME="/path/to/zhouyi_model_zoo"
```

- Set the environment variable `ZHOUYI_DATASET_HOME` to tell python where to find the dataset.

```shell
Since the dataset is very large, we do not provide it in the source release pack.
So you need to download the following datasets and extract them to the same directory.
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    http://images.cocodataset.org/zips/val2017.zip
    http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    https://www.image-net.org/download.php
For example, after downloading the dataset, extract it to `/path/to/dataset`, then set `ZHOUYI_DATASET_HOME` to `/path/to/dataset`
    export ZHOUYI_DATASET_HOME="/path/to/dataset"
```

And also make sure you have installed following packages successfully.
- NPU NN compiler
- NPU simulator

For the installation instructions of these packages, please see the *Arm China AI Platform Zhouyi Compass Software Programming Guide*.

If you want to only running tests on SIMULATOR, execute the command:
```shell
pytest -k 'not rpc' -n 4 --dist loadscope aipu/tests/python/e2e
pytest -k 'not rpc' -n 4 --dist loadscope aipu/tests/python/unittest
```

If you want to running tests on RPC, need to set some Variables:
- Set up RPC system, please refer to *aipu/docs/rpc.md*
- Set `AIPU_TVM_RPC_TRACKER_IP`, `AIPU_TVM_RPC_TRACKER_PORT` and `AIPU_TVM_RPC_KEY` envs like below.
```shell
export AIPU_TVM_RPC_TRACKER_IP="192.168.1.0"
export AIPU_TVM_RPC_TRACKER_PORT="9190"
export AIPU_TVM_RPC_KEY="your_key"
```

- Set `AIPU_TVM_DEVICE_COMPILER` like below, path of cross compiler toolchain.
```shell
export AIPU_TVM_DEVICE_COMPILER="/xxx/aarch64-linux-gnu-g++"
```

then execute the command:
```shell
pytest -n 4 --dist loadscope aipu/tests/python/e2e
pytest -n 4 --dist loadscope aipu/tests/python/unittest
```

### Tests of the Apache TVM
Please refer to the Apache TVM *tests*.
