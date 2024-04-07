..  SPDX-License-Identifier: Apache-2.0
    Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.


.. aipucompass:

Compass Apache TVM
==================

The Apache TVM (`<https://tvm.apache.org/>`_) is an open-source machine learning compiler framework for CPUs, GPUs, and accelerators. It aims to enable machine learning engineers to optimize and run computations efficiently on any hardware backend.

**The Zhouyi Compass is integrated with the Apache TVM for Neural Network (NN) model quick support and heterogeneous execution**. Through the Zhouyi Compass integration with the Apache TVM, your NN model can be supported quickly even though the NN compiler of the Zhouyi Compass does not support the ML framework of the NN model yet.

Graph partition and Heterogeneous execution
-------------------------------------------

**The graph partition solution** is based on the Apache TVM Bring Your Own Codegen (BYOC) framework. It splits the NN operators that the Zhouyi NPU supports and forms the NPU sub graphs. The remained operators form sub graphs corresponding specific device supported.

.. image:: ./images/graph_parition_solution.png
  :width: 500px
  :align: center

The runtime will execute the compiled NN model heterogeneously and automatically. **Heterogeneous execution** of the compiled NN model is transparent to the user application logic. The **compiled NN model consists of several Apache TVM runtime modules**, which include multiple **NPU runtime modules, CPU runtime modules, and other device runtime modules**. The output is obtained when the input data flows over all runtime modules. The data movement between different devices is handled by the Apache TVM runtime automatically.

Use Workflow
------------

The workflow of using the Apache TVM contains two parts: compilation and execution.

**The compilation part always runs on the host development environment.** Compilation part contains:

- Partition the NN model to several sub graphs.
- Process the NPU sub graphs through the NPU NN compiler of the Zhouyi Compass.
- Combine results of all sub graphs together through the Apache TVM runtime execution mechanism.

**The result of compilation is a deployable object**: The object not only can be exported and deployed to the device user environment, but also can be run on the host development environment by the Zhouyi NPU simulator directly.

All work of the **execution part** is handled by `class ExecutionEngine`. It hides the concrete executor (for example, graph executor or VM) details, so that the execution part code is the same regardless of which executor is used when compiling the NN model.

The execution process can not only be run locally in the Simulator, but also be conveniently executed on remote real devices through RPC. Through RPC, except the compiled NN model, all other parts are executed on the host development environment, so the complex preprocess or postprocess can be implemented easily.

Supported Features
------------------

- Support models and operators

    - 120+ models
    - 130+ Relay OP
    - Quantized model
    - PyTorch & ONNX QAT model
- Support for automatic partition of NPU subgraphs
- Support heterogeneous execution of NPU, CPU and others
- Support Linux, QNX, and Android OS deployments
- Easily switch between Simulator or RPC remote hardware operation
- Unified interface support for switching Graph Executor or VM without modifying code
- Release Package

    - Binary encryption distribution package
    - Source code release package
    - Out of box example
- User Guide, API, and source level developer documentation
- Support for Bare Metal scenarios
- Support custom operators for the entire link from Relay front-end to NPU
- Quarterly synchronization with the latest official release

Development Guide
-----------------

Below are development guides, you can learn how to build from scratch.

.. toctree::
   :maxdepth: 1

   source_development_guide.md

More Materials
--------------

Below are materials, you can learn other useful functions of the project.

.. toctree::
   :maxdepth: 1

   rpc.md
   fqa.md
   supporting_bare_metal.md
   pipeline.md
