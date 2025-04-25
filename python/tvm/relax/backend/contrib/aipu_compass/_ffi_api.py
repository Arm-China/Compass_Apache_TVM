# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.

"""FFI APIs for AIPU Compass."""
import tvm


tvm._ffi._init_api("aipu_compass", __name__)
