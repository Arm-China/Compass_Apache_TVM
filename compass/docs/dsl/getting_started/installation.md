<!---SPDX-License-Identifier: Apache-2.0-->
<!---Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.-->

# Installation
Compass DSL is packed as the same Python wheel package with Compass Apache TVM, in order to
provide a end-to-end complete operator development solution to easy write, run and test
Compass DSL programs. There are also some dependencies that need to be installed.

## Dependencies
The following lists other Zhouyi Compass packages that are required:
- NPU OpenCL toolchain
- NPU NN compiler
- NPU simulator
- NPU driver and runtime

For the installation instructions of these packages, see the
*Arm China Zhouyi Compass Software Technical Overview*,
*Arm China Zhouyi Compass NN Compiler User Guide*, and
*Arm China Zhouyi Compass Driver and Runtime User Guide*. Among them, only NPU driver and runtime
need to be compiled, all others just need to be installed.

```{note}
After installing these packages, ensure that the required dynamic libraries, e.g., `libaipudrv.so`,
`libaipu_simulator_x2.so`, can be found from environment variable `LD_LIBRARY_PATH`, and the
required tools, e.g., `aipuocc`, `aipugb`, can be found from environment variable `PATH`.

These environment settings are required each time you use Compass DSL, not only during the installation
phase. It is recommended to put these settings in your shell configuration file, like "~/.cshrc".
```

## Pip Install
```{note}
- You need to use a compatible Python version to install WHL.
- You can use the `sudo` or `--user` option when you do not have the root permission.
```

```shell
pip install dlhc-xxx-xxx.whl
```

## Verification
After successful installation of Compass DSL, you can execute the following commands to run a
simple example.

```shell
python3
>>> from tvm.compass.dsl import BuildManager, script as S
>>> exit()

python3 PYTHON_PACKAGE_PATH/tvm/compass/dsl/samples/tutorial_0_quick_start.py
```

```{note}
The placeholder `PYTHON_PACKAGE_PATH` in above commands represents the location where you install
the Compass DSL Python package. In general, it will be something like
`~/.local/lib/python3.8/site-packages`.

To execute the RPC parts of `tutorial_0_quick_start.py`, the RPC relevant environment
needs to be set up correctly first. For details, see
[How to Use RPC](../how_to_guides/how_to_use_rpc.md).
```

There are more [tutorials about writing Compass DSL programs](./tutorials/index.rst), please
feel free to read.
