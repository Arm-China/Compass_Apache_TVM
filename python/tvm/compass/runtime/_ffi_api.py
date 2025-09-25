# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""FFI APIs for runtime part of Zhouyi Compass extension."""
import tvm


tvm.ffi._init_api("compass.runtime", __name__)
