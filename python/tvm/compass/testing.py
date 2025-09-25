# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Zhouyi Compass extension of testing."""
import functools
import traceback


def clear_traceback(func):
    """Decorator used to clear traceback in order to avoid RPC server timeout when the RPC test case
    raise an exception."""

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except Exception as e:  # pylint: disable=invalid-name
            traceback.clear_frames(e.__traceback__)
            raise e
        return ret

    return _wrapper
