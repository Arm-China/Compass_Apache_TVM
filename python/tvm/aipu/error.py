# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""Zhouyi Compass extension of error handling."""


class CompassCodeGenCError(RuntimeError):
    """Error raised when a Compass OpenCL/C source code can't be generated successfully."""


class CompassCompileCError(RuntimeError):
    """Error raised when a Compass OpenCL/C source code can't be compiled successfully."""
