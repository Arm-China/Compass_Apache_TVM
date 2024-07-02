# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=invalid-name
"""Logger of AIPU Compass."""
import os
import sys
import logging
import functools
import time
import absl.logging


__all__ = [
    "set_logger",
    "DEBUG",
    "DEBUG_ONCE",
    "INFO",
    "WARN",
    "ERROR",
    "timer",
]


class AIPULogger:
    """AIPU Logger. default level: INFO, default handler: stream."""

    def __init__(self):
        logging.root.removeHandler(absl.logging._absl_handler)
        logging.addLevelName(logging.WARNING, "WARN")
        absl.logging._warn_preinit_stderr = 0
        logger = logging.getLogger("AIPU")

        # set default level and handler
        default_level = logging.INFO
        logger.setLevel(default_level)
        self.formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(default_level)
        handler.setFormatter(self.formatter)
        logger.addHandler(handler)
        self.logger = logger

    def set_log_level(self, log_level):
        """
        support 4 level: ERROR, WARN, INFO, DEBUG.
        """

        self.logger.setLevel(log_level)
        for h in self.logger.handlers:
            h.setLevel(log_level)

    def set_log_file(self, file_name):
        """set aipu log file"""

        file_path = os.path.abspath(file_name)
        # remove stream handler
        self.logger.info("AIPU log dump to %s", file_path)
        for h in self.logger.handlers:
            self.logger.removeHandler(h)

        # add file handler
        handler = logging.FileHandler(file_path)
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)


_LOGGER = AIPULogger()
_ONCE_LOGS = []


def DEBUG(msg, *args, **kwargs):
    _LOGGER.logger.debug(msg, *args, **kwargs)


def DEBUG_ONCE(msg):
    if msg not in _ONCE_LOGS:
        _LOGGER.logger.debug(msg)
        _ONCE_LOGS.append(msg)


def INFO(msg, *args, **kwargs):
    _LOGGER.logger.info(msg, *args, **kwargs)


def WARN(msg, *args, **kwargs):
    _LOGGER.logger.warning(msg, *args, **kwargs)


def ERROR(msg, *args, **kwargs):
    _LOGGER.logger.error(msg, *args, **kwargs)


def timer(func):
    """Print time of the decorated function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t_start = time.perf_counter()
        func_name = func.__name__
        log_func = DEBUG if func_name.startswith("_") else INFO
        log_func("%s start", func_name)
        ret = func(*args, **kwargs)
        log_func("%s finished, elapsed time: %.2fs", func_name, time.perf_counter() - t_start)

        return ret

    return wrapper


def set_logger(log_file=None, log_level=None):
    """Set log file and log level"""

    if log_file is None:
        log_file = os.environ.get("AIPU_TVM_LOG_FILE", None)
    if log_level is None:
        log_level = os.environ.get("AIPU_TVM_LOG_LEVEL", None)
    if log_file:
        _LOGGER.set_log_file(log_file)
    if log_level:
        _LOGGER.set_log_level(log_level)
