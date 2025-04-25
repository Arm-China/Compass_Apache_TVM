# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=redefined-builtin, wildcard-import
"""TVM Operator Inventory.

TOPI is the operator collection library for TVM, to provide sugars
for constructing compute declaration as well as optimized schedules.

Some of the schedule function may have been specially optimized for a
specific workload.
"""
#
# This file has been modified by Arm China team.
#
import importlib
from tvm._ffi.libinfo import __version__

# error reporting
from .utils import InvalidShapeError

# Ensure C++ schedules get registered first, so python schedules can
# override them.
from . import cpp

from .math import *
from .tensor import *
from .generic_op_impl import *
from .reduction import *
from .transform import *
from .broadcast import *
from .sort import *
from .scatter import *
from .scatter_elements import *
from .sparse_fill_empty_rows import *
from .sparse_reshape import *
from .argwhere import *
from .scan import *
from .einsum import *
from .unique import *
from .searchsorted import *
from .signal import *


class LazyModule:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name):
        if self.module is None:
            self.module = importlib.import_module(self.module_name, __name__)
        return getattr(self.module, name)


generic = LazyModule(".generic")
nn = LazyModule(".nn")
x86 = LazyModule(".x86")
cuda = LazyModule(".cuda")
gpu = LazyModule(".gpu")
arm_cpu = LazyModule(".arm_cpu")
mali = LazyModule(".mali")
bifrost = LazyModule(".bifrost")
intel_graphics = LazyModule(".intel_graphics")
utils = LazyModule(".utils")
rocm = LazyModule(".rocm")
vision = LazyModule(".vision")
image = LazyModule(".image")
sparse = LazyModule(".sparse")
hls = LazyModule(".hls")
random = LazyModule(".random")
hexagon = LazyModule(".hexagon")
adreno = LazyModule(".adreno")
aipu = LazyModule(".aipu")


# not import testing by default
# because testing can have extra deps that are not necessary
# we can import them from test cases explicitly
# from . import testing
