# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
# pylint: disable=unused-import
"""Interface with package AIPUBuilder."""
import sys
import signal
import functools
from tvm import tir, runtime, ir

try:
    from AIPUBuilder.Optimizer.tools.optimizer_forward import OptForward
    from AIPUBuilder import ops as ops_api

    # Scan and register all of Compass Optimizer plugins.
    from AIPUBuilder.Optimizer import plugins as _

    try:
        from AIPUBuilder.Optimizer import plugins_internal as _
    except ImportError:
        pass

    from AIPUBuilder.Optimizer.framework import QUANTIZE_DATASET_DICT as _DATASET_DICT
    from AIPUBuilder.Optimizer.framework import QUANTIZE_METRIC_DICT as _METRIC_DICT
except ImportError as exc:
    raise RuntimeError("The version of AIPUBuilder is incompatible with tvm.") from exc


# Resetting the signals registered by AIPUBuilder.
try:
    signal.signal(signal.SIGABRT, signal.SIG_DFL)
    signal.signal(signal.SIGSEGV, signal.SIG_DFL)
except ValueError:  # ValueError: signal only works in main thread
    pass


# Workaround for the slow OPT on Python3.8.5 CPU environment.
def _limit_intra_op_thread(func):
    import torch  # pylint: disable=import-outside-toplevel

    def _wrapper(*args, **kwargs):
        old_num_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        ret = func(*args, **kwargs)
        torch.set_num_threads(old_num_threads)
        return ret

    return _wrapper


OptForward.forward = _limit_intra_op_thread(OptForward.forward)


# Wrapper for OPS API, Convert TVM-type args to Python's, such as String to str.
def _convert_args(arg):
    if isinstance(arg, tir.IntImm):
        return int(arg)
    elif isinstance(arg, tir.FloatImm):
        return float(arg)
    elif isinstance(arg, runtime.String):
        return str(arg)
    elif isinstance(arg, (list, tuple, ir.Array)):
        return [_convert_args(x) for x in arg]
    return arg


class _WrappedOPSAPI:
    def __getattr__(self, name):
        attr = getattr(ops_api, name)
        if not callable(attr):
            return attr

        @functools.wraps(attr)
        def _wrapper(*args, **kwargs):
            args = [_convert_args(x) for x in args]
            kwargs = {key: _convert_args(val) for key, val in kwargs.items()}
            return attr(*args, **kwargs)

        return _wrapper


ops = _WrappedOPSAPI()


def create_dataset(class_name, data_file, label_file=None):
    dataset_class = _DATASET_DICT[class_name.lower()]
    return dataset_class(data_file, label_file)


def create_metric(class_name, *args):
    metric = _METRIC_DICT[class_name.lower()](*args)
    metric.reset()
    return metric
