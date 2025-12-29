# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""FFI APIs for top part of Zhouyi Compass extension."""
import tvm


tvm.ffi._init_api("compass", __name__)
