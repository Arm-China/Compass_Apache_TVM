..  SPDX-License-Identifier: Apache-2.0
    Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.

Script API
==========

.. automodule:: tvm.compass.dsl.script
   :members: prim_func, macro

.. autoclass:: tvm.tir.Pointer
   :members: is_nullptr, as_ptr, __getitem__, __add__, __radd__, __sub__,
             __lt__, __le__, __gt__, __ge__, __eq__, __ne__
   :exclude-members: __init__, __new__

.. autoclass:: tvm.compass.dsl.build_manager.BuildManager
   :members:

.. autoclass:: tvm.compass.dsl.executor.Executor
   :members:
   :exclude-members: __init__

.. automodule:: tvm.compass.utils
   :members: get_rpc_session

.. automodule:: tvm.compass.dsl.utils
   :members: hw_native_vdtype

.. automodule:: tvm.compass.dsl.testing
   :members: rand

.. automodule:: tvm.compass.dsl.script.ir.ir
   :members:

.. automodule:: tvm.compass.dsl.script.ir.axis
   :members:
