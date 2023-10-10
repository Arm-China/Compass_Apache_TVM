# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.


try:
    from xdist.scheduler.load import LoadScheduling

    HAVE_XDIST = True
except ImportError:
    HAVE_XDIST = False

# Add pytest xdist LoadScheduling here to override TvmTestScheduler which maybe caused test cases fail to run completely.
if HAVE_XDIST:

    def pytest_xdist_make_scheduler(config, log):
        return LoadScheduling(config, log)


def pytest_configure(config):
    marker_list = ["X2_1204"]
    for marker in marker_list:
        config.addinivalue_line("markers", marker)
