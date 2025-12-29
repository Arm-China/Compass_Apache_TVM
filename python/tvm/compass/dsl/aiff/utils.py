# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Utilities for AIFF APIs."""


def align8(x):
    return ((x + 7) >> 3) << 3
