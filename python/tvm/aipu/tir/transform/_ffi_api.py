# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""FFI APIs for TIR transform part of Zhouyi Compass extension."""
import tvm


tvm._ffi._init_api("aipu.tir.transform", __name__)
