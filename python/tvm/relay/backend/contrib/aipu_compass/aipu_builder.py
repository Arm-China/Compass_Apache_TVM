# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=unused-import
"""Interface with package AIPUBuilder."""
import os
import re
import sys
import signal
import contextlib
from subprocess import run, STDOUT
from multiprocessing import Process

try:
    from AIPUBuilder._C._core import redirect_stdout as _redirect_std
    from AIPUBuilder._C._tools import aipugb as _aipugb_pyapi
    from AIPUBuilder._C._tools import aipurun as _aipurun_pyapi
    from AIPUBuilder.Optimizer.tools.optimizer_main import main as _opt_main
    from AIPUBuilder.Optimizer.tools.optimizer_forward import OptForward
    from AIPUBuilder.simplifier.main import main as _gsim_main
    from AIPUBuilder.Profiler.main import main as _profiler_main

    # Scan and register all of AIPU Optimizer plugins.
    from AIPUBuilder.Optimizer import plugins as _
    from AIPUBuilder.Optimizer.framework import QUANTIZE_DATASET_DICT as _DATASET_DICT
    from AIPUBuilder.Optimizer.framework import QUANTIZE_METRIC_DICT as _METRIC_DICT
except ImportError:
    from tvm.aipu import logger

    logger.ERROR(
        "The verion of AIPUBuilder is incompatible with tvm. "
        "Please use appropriate AIPUBuilder package."
    )
    sys.exit(1)


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


@contextlib.contextmanager
def redirect_std(out, err=None):
    """Context manager to redirect standard output and error for tools of AIPUBuilder."""
    if err is None:
        err = out
    sys.stdout.flush()
    sys.stderr.flush()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    _redirect_std(True)
    yield
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout, sys.stderr = old_out, old_err
    _redirect_std(False)


def _aipuopt_pyapi(sys_argv):
    old_sys_argv = sys.argv
    sys.argv = sys_argv
    try:
        ret = _opt_main()
    except Exception as e:  # pylint: disable=broad-except, invalid-name
        ret = 1
        sys.stderr.write(f"{e}")
    sys.argv = old_sys_argv
    return ret


def _aipugsim_pyapi(sys_argv):
    old_sys_argv = sys.argv
    sys.argv = sys_argv
    try:
        ret = _gsim_main()
    except Exception as e:  # pylint: disable=broad-except, invalid-name
        ret = 1
        sys.stderr.write(f"{e}")
    sys.argv = old_sys_argv
    return ret


def _aipuprofiler_pyapi(sys_argv):
    old_sys_argv = sys.argv
    sys.argv = sys_argv
    try:
        ret = _profiler_main()
    except Exception as e:  # pylint: disable=broad-except, invalid-name
        ret = 1
        sys.stderr.write(f"{e}")
    sys.argv = old_sys_argv
    return ret


_NAME2PYAPI_FUNC = {
    "aipuopt": ("Optimizer", _aipuopt_pyapi),
    "aipugb": ("GBuilder", _aipugb_pyapi),
    "aipugsim": ("GSim", _aipugsim_pyapi),
    "aipurun": ("AIPURun", _aipurun_pyapi),
    "aipu_profiler": ("Profiler", _aipuprofiler_pyapi),
}


def check_call_aipu_tool(cmd, work_dir=os.getcwd(), by_subprocess=True):
    """Call tools of AIPUBuilder through Python API and check the return code."""
    work_dir = os.path.abspath(work_dir)
    old_cwd = os.getcwd()
    if work_dir != old_cwd:
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)

    exe_name = cmd[0]
    tool_name, pyapi_func = _NAME2PYAPI_FUNC[exe_name]
    log_file = f"{work_dir}/{exe_name}.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Command Line: {' '.join(cmd)}\n")
        f.flush()
        if by_subprocess:
            env = None
            if exe_name == "aipuopt":
                # Workaround for the slow OPT on Python3.8.5 CPU environment.
                env = dict(os.environ)
                env["OMP_NUM_THREADS"] = "4"

            ret_code = run(
                cmd,
                stdout=f,
                stderr=STDOUT,
                check=False,
                encoding="utf-8",
                env=env,
                universal_newlines=True,
            ).returncode
        else:
            with redirect_std(f):
                p = Process(target=lambda: pyapi_func(cmd))
                p.start()
                p.join()
                ret_code = p.exitcode

    count_errors = 0
    with open(log_file, "r", encoding="utf-8") as f:
        error_pattern = re.compile(r"(?<=Total errors: )\d+")
        for line in f.readlines():
            digit_list = error_pattern.findall(line)
            if len(digit_list) == 0:
                continue
            for digit in digit_list:
                if int(digit) > 0:
                    count_errors = int(digit)
                    break
            if count_errors != 0:
                break

    if old_cwd != os.getcwd():
        os.chdir(old_cwd)

    if ret_code != 0 or count_errors != 0:
        raise RuntimeError(
            f"Error happened when executing the AIPU {tool_name}, for more details, please refer to"
            f' the log file "{log_file}".'
        )


def create_dataset(class_name, data_file, label_file=None):
    dataset_class = _DATASET_DICT[class_name.lower()]
    return dataset_class(data_file, label_file)


def create_metric(class_name, *args):
    metric = _METRIC_DICT[class_name.lower()](*args)
    metric.reset()
    return metric
