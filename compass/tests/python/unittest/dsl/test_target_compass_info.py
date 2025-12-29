# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
import pytest
from tvm.compass import CompassInfo


@pytest.mark.X2
def test_compass_info():
    info = CompassInfo.get("X2_1204")
    assert info is not None
    assert info.core_count == 1
    assert info.tec_count == 4
    assert info.version == "X2"
    assert info.vector_width == 256
    assert info.lsram_size() == (32 * 1024)
    assert info.gsram_size() == (256 * 1024)


if __name__ == "__main__":
    test_compass_info()
