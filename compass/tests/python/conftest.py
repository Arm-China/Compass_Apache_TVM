# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.


def pytest_configure(config):
    marker_list = ["X1", "X2", "X3P", "NOT_X1", "NOT_X2", "NOT_X3P", "REQUIRE_RPC"]
    for marker in marker_list:
        config.addinivalue_line("markers", marker)
