..  SPDX-License-Identifier: Apache-2.0
    Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.

Script API
==========

.. automodule:: tvm.aipu.script
   :members: prim_func, macro

.. autoclass:: tvm.tir.Pointer
   :members: is_nullptr, as_ptr, __getitem__, __add__, __radd__, __sub__,
             __lt__, __le__, __gt__, __ge__, __eq__, __ne__
   :exclude-members: __init__, __new__

.. autoclass:: tvm.aipu.tir.BuildManager
   :members:

.. autoclass:: tvm.aipu.tir.executor.Executor
   :members:
   :exclude-members: __init__

.. automodule:: tvm.aipu.utils
   :members: rand, get_rpc_session, hw_native_vdtype

.. automodule:: tvm.aipu.script.ir.ir
   :members:

.. automodule:: tvm.aipu.script.ir.axis
   :members:
