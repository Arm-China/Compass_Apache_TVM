# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Zhouyi Compass DSL extension of testing."""
import os
import sys
import re
from contextlib import contextmanager
import numpy as np
from ... import DataType, get_range, testing
from ..logger import INFO
from ..testing import clear_traceback  # pylint: disable=unused-import
from ..utils import check_call_compass_tool, control_option
from .utils import is_hw_native_scalar_dtype, HW_NATIVE_SCALAR_DTYPES


def rand(shape, dtype, low=None, high=None, enable_corner_values=True, return_python_type=False):
    """Random values in a given shape, dtype and [low, high) range (including low, excluding high).

    Parameters
    ----------
    shape : Union[int, Tuple[int], List[int]]
        The element number on which rand is performed.

    dtype : str
        The data type.

    low : Optional[int, float]
        The minimum threshold for the rand range.

    high : Optional[int, float]
        The maximum threshold for the rand range.

    enable_corner_values : Optional[bool]
        Whether the corner values are forced to be included.
        Note:
        1. The corner values contain: low or dtype minimum value, high or dtype maximum value,
        and zero value when zero is in the random range.
        2. When the value is True and the number of elements is less than the number of corner
        values, it is uncertain whether corner values are forced to be included: the existence of
        corner values depends on randomness.

    return_python_type : Optional[bool]
        Whether return the result as Python native type or not, if it is False, the result are
        returned as NumPy type.

    Returns
    -------
    out: Union[float, int, List[float], List[int], numpy.ndarray]
        Rand values, scalar when shape is 1 or numpy.ndarray when shape is a tuple of int.

    Examples
    --------
    .. code-block:: python

        # Generate NumPy objects.
        ndarray_i8_a = rand(100, "int8")
        ndarray_fp16_b = rand((4, 16), "float16", low=-100, high=100)
        ndarray_int16_c = rand((1,), "int16")
        numpy_fp32_c = rand(1, low=0, "float32")

        # Generate Python native type objects.
        float_list_d = rand((2, 30), "float32", high=5.5, return_python_type=True)
        int_value_e = rand(1, "int32", enable_corner_values=False, return_python_type=True)
        int_list_f = rand((1,), "int8", return_python_type=True)

    """
    if not control_option.random_seed_is_set:
        seed = control_option.random_seed or np.random.randint(0, 2**31)
        np.random.seed(seed)
        control_option.random_seed_is_set = True
        INFO(f'Reproduce with the random seed by "setenv CPS_DSL rs={seed}".')

    dtype = DataType(dtype)
    msg = f'The arg "dtype" expect one of {HW_NATIVE_SCALAR_DTYPES}, but got: "{dtype}".'
    assert is_hw_native_scalar_dtype(dtype), msg

    if dtype.is_bool:
        out = np.random.uniform(size=shape) < 0.5
        out = out[0] if shape == 1 else out
        return out.tolist() if return_python_type else out

    minv, maxv = get_range(dtype)
    minv = minv if low is None else low
    maxv = maxv if high is None else (high - (1e-5 if dtype.is_float else 1))
    if minv == maxv:
        return getattr(np, dtype)(minv) if shape == 1 else np.full(shape, minv, dtype=dtype)

    assert minv < maxv
    if dtype.is_floating:
        # Use normal distribution to generate more elements with decimal part.
        std_dev = min((float(maxv) - minv) / 6, 700 if dtype.is_float16 else 1e6)
        mean = (float(minv) + maxv) / 2
        out = np.random.normal(mean, std_dev, shape).clip(minv, maxv).astype(dtype)
    else:
        out = np.random.randint(minv, maxv, shape, dtype)

    corner_values = (minv, 0, maxv) if minv < 0 < maxv else (minv, maxv)
    if enable_corner_values and out.size > len(corner_values):
        occupied_indices = []
        for val in corner_values:
            idx = tuple(np.random.randint(out.shape))
            while idx in occupied_indices:
                idx = tuple(np.random.randint(out.shape))

            out[idx] = val
            occupied_indices.append(idx)

    out = out[0] if shape == 1 else out
    return out.tolist() if return_python_type else out


def _binary_equal(x, y):
    """Make sure the binaries cannot differ by more than 1."""
    if not (x.dtype == y.dtype and x.dtype in (np.float16, np.float32, np.dtype("bfloat16"))):
        return False

    if x.dtype in (np.float16, np.dtype("bfloat16")):
        view_dtype = np.int16
    else:
        view_dtype = np.int32

    finite_mask = np.isfinite(x) & np.isfinite(y)
    return np.isclose(x.view(view_dtype), y.view(view_dtype), rtol=0, atol=2.1) & finite_mask


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
        if not (atol == 0 and rtol == 0):
            is_close = _binary_equal(actual, desired)
            actual[is_close] = desired[is_close]

    try:
        if actual.dtype == "bfloat16":
            actual = actual.astype("float32")
            desired = desired.astype("float32")
        testing.assert_allclose(actual, desired, rtol, atol)
    except AssertionError as exc:
        not_close = ~np.isclose(actual, desired, rtol=rtol, atol=atol, equal_nan=True)
        raise AssertionError(
            f"{exc}\nMismatched x:\n{actual[not_close]}\nMismatched y:\n{desired[not_close]}"
        ) from None


@contextmanager
def _log_block(start_msg, end_msg):
    sys.stdout.write(start_msg + "\n")
    sys.stdout.flush()
    try:
        yield
    finally:
        sys.stdout.write(end_msg + "\n")
        sys.stdout.flush()


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
    from AIPUBuilder.executor import GtForward  # pylint: disable=import-outside-toplevel

    files = os.listdir(case_file_path)
    for file_str in ("graph.def", "weight.bin", "input0.bin"):
        assert file_str in files, f"Not found {file_str} in {case_file_path}."
    graph = os.path.join(case_file_path, "graph.def")
    weight = os.path.join(case_file_path, "weight.bin")
    inputs = ",".join([os.path.join(case_file_path, i) for i in files if "input" in i])
    stage_info = lambda x: f"[DSL OP Test]: {'='*20}Stage {x}{'='*20}"

    # Run DSL OP integration test through "aipurun"
    with _log_block(stage_info("1(aipurun) Start"), stage_info("1(aipurun) End")):
        cmd = ["aipurun", graph, "-w", weight, "-i", inputs, "--target", target]
        cmd += ["--set_asid", "asid0=0x100000000,asid1=0x200000000"]
        # disable passes
        passes = ""
        disable_pass_file = os.getenv("CPS_TVM_DISABLE_PASS_FILE")
        if disable_pass_file and "tvm_update_aipubuilder" not in os.getenv("JOB_NAME", ""):
            assert os.path.isfile(disable_pass_file), f"File not exists: {disable_pass_file}."
            with open(disable_pass_file, "r") as p:
                passes = p.read().strip()
        else:
            cmd_str = "aipurun --show-all-passes"
            passes_list = list()
            with os.popen(cmd_str, "r") as p:
                pattern = r"PassTuple\((.*?)\)"
                for line in p.readlines():
                    match = re.search(pattern, line)
                    if match:
                        extracted_string = match.group(1)
                        passes_list += extracted_string.split(",")
                enabled_passes = ("DataLayoutSchedule",)
                passes = ",".join([p for p in passes_list if p not in enabled_passes])
        if passes:
            cmd += ["--disable-pass", passes]

        # if "aipu_simulator_xx" not in PATH, "aipu_simulator_path" will be an empty str
        cmd_str = f"which aipu_simulator_{target.split('_')[0].lower()}"
        simulator_path = os.popen(cmd_str).read().strip()
        assert simulator_path, f"Please add simulator path of target '{target}' to the ENV 'PATH'."
        cmd += ["--simulator", simulator_path]

        # run aipurun
        check_call_compass_tool(cmd, work_dir=work_dir)
        assert "output.bin" in os.listdir(work_dir), "can not found output!"

    # Run Gt
    with _log_block(stage_info("2(executor) Start"), stage_info("2(executor) End")):
        cur_dir = os.getcwd()
        try:
            os.chdir(work_dir)
            exeutor = GtForward(graph, weight, inputs=inputs, target=target, disable_pass=passes)
            gts = exeutor.forward()
        finally:
            os.chdir(cur_dir)

    # Check Results
    with _log_block(stage_info("3(compare) Start"), stage_info("3(compare) End")):
        for i, gt_out in enumerate(gts):
            out_bin = "output.bin" if i == 0 else f"output.bin{i}"
            out_bin = os.path.join(work_dir, out_bin)
            output = np.fromfile(out_bin, dtype=gt_out.dtype).reshape(gt_out.shape)
            assert_allclose(output, gt_out, atol=1e-3)
