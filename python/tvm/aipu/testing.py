# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
"""Zhouyi Compass extension of testing."""
import numpy as np
from .. import testing


def assert_allclose(actual, desired, rtol=1e-7, atol=1e-7):
    """Simple wrapper of the corresponding API of TVM Testing."""
    assert (
        actual.dtype == desired.dtype
    ), f'Argument type mismatch: 0-th: "{actual.dtype}" vs. 1-th: "{desired.dtype}".'
    if isinstance(actual.dtype, np.integer):
        rtol, atol = 0, 0
    testing.assert_allclose(actual, desired, rtol, atol)
