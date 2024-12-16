# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Utility to invoke Compass OpenCL compiler found in the system."""
import os
import shlex
import shutil
import tempfile
from subprocess import run, PIPE, STDOUT
from ..utils import abspath, canonicalize_target, control_option
from ..error import CompassCompileCError


INCLUDE_DIR_IN_PKG = abspath(f"{__file__}/../../include")


def compile_c(
    code,
    target="X2_1204",
    extra_options="",
    output_path=None,
    work_dir=None,
    save_log=True,
):
    """Compile Compass OpenCL source code through "aipuocc".

    Parameters
    ----------
    code : str
        The path or the content string of the Compass OpenCL source code file.

    target : Union[str, tvm.target.Target]
        The target that the code is compiled for. It can be a literal target string or a
        tvm.target.Target object.

    extra_options : str
        The extra compilation options need to be passed to "aipuocc".

    output_path : Optional[str]
        The path of the compilation result file, if set, the compilation result will be written to
        the specified path instead of a temporary file.

    work_dir : Optional[str]
        The working directory in where to execute the compilation command, if not set, the current
        working directory will be used.

    save_log : bool
        Whether save the compilation log to disk or not, if set to True, the compilation log will be
        saved as a file whose path is same as the compilation result file except the file extension.

    Return
    ------
    ret : bytearray
        The content of the compilation result file.
    """
    target = canonicalize_target(target)
    temp_dir = tempfile.mkdtemp()

    out_path = abspath(output_path) or os.path.join(temp_dir, "a.o")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if "\n" not in code:  # Simple way to consider "code" is a path.
        code_path = abspath(code)
    else:
        code_path = abspath(f"{os.path.splitext(out_path)[0]}.cl")
        open(code_path, "w", encoding="utf-8").write(code)

    work_dir = abspath(work_dir) or os.getcwd()
    os.makedirs(work_dir, exist_ok=True)

    if code_path.startswith(work_dir):
        code_path = os.path.relpath(code_path, work_dir)
    if out_path.startswith(work_dir):
        out_rel_path = os.path.relpath(out_path, work_dir)

    cmd = ["aipuocc", "-c", "-target", "aipu", f"-mcpu={target.mcpu.split('MP')[0]}", "-I"]
    cmd += [INCLUDE_DIR_IN_PKG]
    cmd += shlex.split(extra_options)
    cmd += ["-o", out_rel_path, code_path]

    result = run(cmd, stdout=PIPE, stderr=STDOUT, cwd=work_dir, check=False, encoding="utf-8")
    log_str = f"Current Working Directory: {work_dir}\nCommand Line: {' '.join(result.args)}\n"
    log_str += f"Standard Output & Error:\n{result.stdout}"

    if save_log:
        log_path = abspath(f"{os.path.splitext(out_path)[0]}.log")
        open(log_path, "w", encoding="utf-8").write(log_str)

    if result.returncode != 0:
        if len(os.listdir(temp_dir)) == 0:
            os.rmdir(temp_dir)
        msg = "Compass OpenCL Compilation Error:\n"
        if save_log:
            msg += f"Log File Path: {log_path}\n"
        raise CompassCompileCError(msg + log_str)

    if control_option.is_asm:
        cmd[-2] = f"{os.path.splitext(out_rel_path)[0]}.s"
        cmd.insert(-3, "-S")
        run(cmd, cwd=work_dir, check=True, encoding="utf-8")

    data = bytearray(open(out_path, "rb").read())
    assert data, "Compass OpenCL compilation result is empty."

    # Only delete the temporary directory when the code is compiled successfully.
    shutil.rmtree(temp_dir, ignore_errors=True)
    return data
