# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=unused-import
"""Interface with package AIPUBuilder."""
import sys
import signal

try:
    from AIPUBuilder.Optimizer.tools.optimizer_forward import OptForward

    # Scan and register all of AIPU Optimizer plugins.
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
signal.signal(signal.SIGABRT, signal.SIG_DFL)
signal.signal(signal.SIGSEGV, signal.SIG_DFL)


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


def create_dataset(class_name, data_file, label_file=None):
    dataset_class = _DATASET_DICT[class_name.lower()]
    return dataset_class(data_file, label_file)


def create_metric(class_name, *args):
    metric = _METRIC_DICT[class_name.lower()](*args)
    metric.reset()
    return metric
