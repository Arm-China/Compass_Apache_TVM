# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Zhouyi Compass extension of error handling."""


class CompassCodeGenCError(RuntimeError):
    """Error raised when a Compass OpenCL/C source code can't be generated successfully."""


class CompassCompileCError(RuntimeError):
    """Error raised when a Compass OpenCL/C source code can't be compiled successfully."""
