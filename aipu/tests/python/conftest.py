# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.


def pytest_configure(config):
    marker_list = ["X2_1204"]
    for marker in marker_list:
        config.addinivalue_line("markers", marker)
