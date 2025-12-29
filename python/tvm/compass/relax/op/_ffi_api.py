# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
# pylint: disable=unused-argument
"""Constructor APIs"""
import tvm.ffi

tvm.ffi._init_api("relax.op.compass", __name__)
