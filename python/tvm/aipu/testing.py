# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Zhouyi Compass extension of testing."""
import os
import sys
import numpy as np
from AIPUBuilder.executor import GtForward
from .. import testing
from .utils import check_call_aipu_tool


def _binary_equal(x, y):
    """Make sure the binaries cannot differ by more than 1."""
    if not (x.dtype == y.dtype and x.dtype in (np.float16, np.float32)):
        return False

    if x.dtype == np.float16:
        view_dtype = np.int16
    else:
        view_dtype = np.int32

    finite_mask = np.isfinite(x) & np.isfinite(y)
    return np.isclose(x.view(view_dtype), y.view(view_dtype), rtol=0, atol=1.1) & finite_mask


def assert_allclose(actual, desired, rtol=None, atol=None):
    """Simple wrapper of the corresponding API of TVM Testing."""
    actual = np.array(actual)
    desired = np.array(desired)
    assert (
        actual.dtype == desired.dtype
    ), f'Argument type mismatch: 0-th: "{actual.dtype}" vs. 1-th: "{desired.dtype}".'

    rtol = 0 if rtol is None else rtol
    if np.issubdtype(actual.dtype, np.integer):
        atol = 0 if atol is None else atol
    else:
        atol = 1e-6 if atol is None else atol
        is_close = _binary_equal(actual, desired)
        actual[is_close] = desired[is_close]

    try:
        testing.assert_allclose(actual, desired, rtol, atol)
    except AssertionError as exc:
        not_close = ~np.isclose(actual, desired, rtol=rtol, atol=atol, equal_nan=True)
        raise AssertionError(
            f"{exc}\nMismatched x:\n{actual[not_close]}\nMismatched y:\n{desired[not_close]}"
        ) from None


def run_op_case(work_dir, case_file_path, target):
    """
    Parameters
    ----------
    work_dir : str
        the dir in which all operations in this function are executed.

    case_file_path : str
        the dir that include all the file aipurun need.
        the files is graph.def, weight.bin, input0.bin, input1.bin ...

    target : str

    """
    files = os.listdir(case_file_path)
    for file_str in ("graph.def", "weight.bin", "input0.bin"):
        assert file_str in files, f"Not found {file_str} in {case_file_path}."
    graph = os.path.join(case_file_path, "graph.def")
    weight = os.path.join(case_file_path, "weight.bin")
    inputs = ",".join([os.path.join(case_file_path, i) for i in files if "input" in i])

    def flush_out(message):
        sys.stdout.write(message + "\n")
        sys.stdout.flush()

    stage_info = lambda x: flush_out(f"[DSL OP Test]: {'='*20}Stage {x}{'='*20}")
    stage_info("1(aipurun) Start")
    # Run DSL OP
    cmd = ["aipurun", graph, "-w", weight, "-i", inputs, "--target", target]
    # disable passes
    passes = ""
    disable_pass_file = os.environ.get("AIPU_TVM_DISABLE_PASS_FILE", None)
    if disable_pass_file and "tvm_update_aipubuilder" not in os.environ.get("JOB_NAME", ""):
        assert os.path.isfile(disable_pass_file), f"File not exists: {disable_pass_file}."
        with open(disable_pass_file, "r") as p:
            passes = p.read().strip()
    else:
        show_passes = "aipurun --show-all-passes"
        try:
            with os.popen(show_passes, "r") as p:
                all_passes = p.readlines()
                passes = ",".join(["".join(i[10:-2]) for i in [all_passes[2], all_passes[5]]])
        except IndexError:
            print(f"[WARN] Failed to get passes through '{show_passes}', not disable any passes.")

    if passes:
        cmd += ["--disable-pass", passes]
    # run aipurun
    check_call_aipu_tool(cmd, work_dir=work_dir)
    assert "output.bin" in os.listdir(work_dir), "can not found output!"
    stage_info("1(aipurun) End")
    stage_info("2(executor) Start")
    # Run Gt
    cur_dir = os.getcwd()
    try:
        os.chdir(work_dir)
        exeutor = GtForward(graph, weight, inputs=inputs, target=target, disable_pass=passes)
        gts = exeutor.forward()
    finally:
        os.chdir(cur_dir)
    stage_info("2(executor) End")
    stage_info("3(compare) Start")
    if (
        os.getenv("AIPU_TVM_DEV_OP_LIB_IMPL_ID") == "0"
        and case_file_path.split("/")[-2] == "eltwise"
    ):
        return
    # Check Results
    for i, gt_out in enumerate(gts):
        out_bin = "output.bin" if i == 0 else f"output.bin{i}"
        out_bin = os.path.join(work_dir, out_bin)
        output = np.fromfile(out_bin, dtype=gt_out.dtype).reshape(gt_out.shape)
        assert_allclose(output, gt_out, atol=1e-3)
