# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""FFI APIs for script part of the Zhouyi Compass extension."""
import tvm


tvm._ffi._init_api("aipu.script", __name__)
