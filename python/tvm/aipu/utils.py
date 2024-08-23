# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Common AIPU utilities."""
import os
import re
import operator
import functools
from subprocess import run, STDOUT
import numpy as np
from .. import autotvm, contrib, rpc, tir, target as tgt, DataType, get_range, int_within_range
from .logger import INFO


HW_NATIVE_STORAGE_DTYPES = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float16")
HW_NATIVE_STORAGE_DTYPES += ("float32",)
_HW_NATIVE_SCALAR_DTYPES = HW_NATIVE_STORAGE_DTYPES + ("bool",)
HW_NATIVE_VDTYPES = ("int8x32", "uint8x32", "int16x16", "uint16x16", "int32x8", "uint32x8")
HW_NATIVE_VDTYPES += ("float16x16", "float32x8")
HW_NATIVE_MASK_TYPES = ("boolx8", "boolx16", "boolx32")
VALID_ADDR_DTYPES = HW_NATIVE_STORAGE_DTYPES + HW_NATIVE_VDTYPES + ("float32x16",)


def is_hw_native_scalar_dtype(dtype):
    return str(dtype) in _HW_NATIVE_SCALAR_DTYPES


def is_hw_native_dtype(dtype):
    return str(dtype) in (_HW_NATIVE_SCALAR_DTYPES + HW_NATIVE_VDTYPES + HW_NATIVE_MASK_TYPES)


_EXE_NAME2TOOL_NAME = {
    "aipuopt": "Optimizer",
    "aipugb": "GBuilder",
    "aipugsim": "GSim",
    "aipurun": "AIPURun",
    "aipu_profiler": "Profiler",
}


def check_call_aipu_tool(cmd, work_dir=os.getcwd()):
    """Call tools of AIPUBuilder through sub process and check the return code."""
    work_dir = os.path.abspath(work_dir)
    old_cwd = os.getcwd()
    if work_dir != old_cwd:
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)

    exe_name = cmd[0]
    log_file = f"{work_dir}/{exe_name}.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Command Line: {' '.join(cmd)}\n")
        f.flush()
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
            text=True,
        ).returncode

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
            f"Error happened when executing the AIPU {_EXE_NAME2TOOL_NAME[exe_name]}, for more "
            f'details, please refer to the log file "{log_file}".'
        )


def abspath(path, base_dir=None):
    """Return the absolute path of the given path and the base directory.

    Parameters
    ----------
    path : Optional[str]
        The given path.

    base_dir : Optional[str]
        The base directory will be used only when the given path is a relative
        one, if it is None, the current working directory will be used.

    Return
    ------
    ret : Optional[str]
        The result absolute path. None if the given path is None.
    """
    if path is None:
        return None

    path = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(path):
        # "abspath" here is used to remove "." and "..".
        path = os.path.abspath(path)
    else:
        path = os.path.abspath(f"{base_dir or os.getcwd()}/{path}")
    return path


def get_rpc_session(
    session_timeout=600, rpc_key=None, tracker_host=None, tracker_port=None, priority=1
):
    """Connect to the RPC tracker and get an RPC session with the RPC key.

    Parameters
    ----------
    session_timeout : Optional[float]
        The duration of the session, which allows the server to kill
        the connection when duration is longer than this value.
        When duration is zero, it means that the request must always be kept alive.

    rpc_key : Optional[str]
        The type key of the device.
        If rpc_key = "None", get it from env "AIPU_TVM_RPC_KEY".

    tracker_host : Optional[str]
        The hostname or IP address of the RPC tracker.
        If tracker_host = "None", get it from env "AIPU_TVM_RPC_TRACKER_IP".

    tracker_port: Optional[int, str]
        The port of the RPC tracker.
        If tracker_port = "None", get it from env "AIPU_TVM_RPC_TRACKER_PORT".

    priority : Optional[int]
        The priority of the request.
        If priority = "None", get it from env "AIPU_TVM_RPC_PRIORITY".

    Returns
    -------
    sess : tvm.rpc.RPCSession
        The RPC session that is already connected to the RPC server.
    """
    # Override logic of RPC key is special, function argument has higher priority.
    rpc_key = rpc_key or os.getenv("AIPU_TVM_RPC_KEY")
    assert rpc_key, 'Set RPC key through arg or env "AIPU_TVM_RPC_KEY".'

    tracker_host = os.getenv("AIPU_TVM_RPC_TRACKER_IP") or tracker_host
    assert tracker_host, 'Set RPC tracker host through arg or env "AIPU_TVM_RPC_TRACKER_IP".'
    tracker_port = os.getenv("AIPU_TVM_RPC_TRACKER_PORT") or tracker_port
    assert tracker_port, 'Set RPC tracker port through arg or env "AIPU_TVM_RPC_TRACKER_PORT".'
    priority = os.getenv("AIPU_TVM_RPC_PRIORITY") or priority
    assert priority, 'Set RPC priority through arg or env "AIPU_TVM_RPC_PRIORITY".'

    valid_rpc_keys = os.getenv("AIPU_TVM_VALID_RPC_KEYS")
    if valid_rpc_keys:
        valid_rpc_keys = tuple(x.strip() for x in valid_rpc_keys.split("|") if x.strip() != "")
        assert (
            rpc_key in valid_rpc_keys
        ), f"Invalid RPC key '{rpc_key}', the valid choices are {valid_rpc_keys}."

    return rpc.connect_tracker(tracker_host, int(tracker_port)).request(
        key=rpc_key, priority=int(priority), session_timeout=session_timeout
    )


def check_remote(rpc_key=None, tracker_host=None, tracker_port=None):
    """Check the remote device is available or not."""
    pool = contrib.popen_pool.PopenPoolExecutor(max_workers=1, timeout=10)

    def _check():
        get_rpc_session(5, rpc_key, tracker_host, tracker_port, 100)

    try:
        pool.submit(_check).result()
    except TimeoutError:
        return False
    return True


def sync_compass_output_dir(rpc_sess, filter_fn=lambda x: True):
    """Synchronize files of compass output directory on RPC server to local.

    Parameters
    ----------
    rpc_sess : tvm.rpc.RPCSession
        The RPC session that is already connected to the RPC server.

    filter_fn : Optional[Callable[[str], bool]]
        The function used to select the files that need to be synchronized to local. It will be
        called for each file, only the files whose return value are True will be selected.
    """
    from tvm.relay.backend.contrib import (  # pylint: disable=import-outside-toplevel
        aipu_compass,
    )

    err_msg = f'The arg "rpc_sess" expect a RPC session, but got: "{type(rpc_sess)}".'
    assert isinstance(rpc_sess, rpc.RPCSession), err_msg

    remote_files = tuple(
        x for x in rpc_sess.list_files(".") if x.startswith("compass_output") and filter_fn(x)
    )
    local_output_dir = aipu_compass.AipuCompassBasicConfig.get().common["output_dir"]

    for remote_file in remote_files:
        rel_path = remote_file.split(os.path.sep, 1)[1]
        open(f"{local_output_dir}/{rel_path}", "wb").write(rpc_sess.download(remote_file))
        INFO(f'Downloaded "{rel_path}" into "{local_output_dir}".')


def prod_const(arr):
    """Reduce product the given input sequence to a constant value."""
    const_arr = []
    for x in arr:
        if isinstance(x, tir.IterVar):
            x = x.dom.extent
        const_arr.append(autotvm.utils.get_const_int(x))

    return functools.reduce(operator.mul, const_arr, 1)


def canonicalize_target(target):
    """Canonicalize target and return tvm.target.Target."""
    if isinstance(target, tgt.Target):
        return target
    assert isinstance(target, str), f"Unsupported target type: {type(target)}."
    if not target.startswith("aipu"):
        target = "aipu -mcpu=" + target
    return tgt.Target(target)


def hw_native_vdtype(dtype):
    """Get the corresponding hardware native vector data type.

    Parameters
    ----------
    dtype : Union[str, DataType]
        The given data type, can be any scalar or vector data type except boolean ones.

    Returns
    -------
    ret: DataType
        The corresponding hardware native vector data type.

    Examples
    --------
    .. code-block:: python

        # Generate from string objects.
        i8x32 = hw_native_vdtype("int8")
        fp32x8 = hw_native_vdtype("float32")

        # Generate from DataType objects.
        u16x16 = hw_native_vdtype(DataType("uint16"))
        i32x8 = hw_native_vdtype(DataType("int32"))

    See Also
    --------
    - :doc:`../../language_basics/types`
    """
    scalar_dtype = DataType(dtype)
    assert not scalar_dtype.is_bool, "Does not support boolean data type."
    return scalar_dtype.with_lanes(256 // scalar_dtype.bits)


# Don't set value here, set it through environment variable "AIPU_TVM_RANDOM_SEED".
_RANDOM_SEED = None


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
    global _RANDOM_SEED
    if _RANDOM_SEED is None:
        _RANDOM_SEED = os.getenv("AIPU_TVM_RANDOM_SEED") or np.random.randint(0, 2**31)
        np.random.seed(int(_RANDOM_SEED))
        INFO(f'Reproduce with the random seed by "setenv AIPU_TVM_RANDOM_SEED {_RANDOM_SEED}".')

    err_msg = f'The arg "dtype" expect one of {_HW_NATIVE_SCALAR_DTYPES}, but got: "{dtype}".'
    assert is_hw_native_scalar_dtype(dtype), err_msg
    dtype_str = dtype
    dtype = DataType(dtype)

    if dtype.is_bool:
        out = np.random.uniform(size=shape) < 0.5
        out = out[0] if shape == 1 else out
        return out.tolist() if return_python_type else out

    minv, maxv = get_range(dtype)
    minv = minv if low is None else low
    maxv = maxv if high is None else (high - (1e-5 if dtype.is_float else 1))
    if minv == maxv:
        return getattr(np, dtype_str)(minv) if shape == 1 else np.full(shape, minv, dtype=dtype_str)

    assert minv < maxv
    if dtype.is_float:
        # Use normal distribution to generate more elements with decimal part.
        std_dev = min((float(maxv) - minv) / 6, 1e6 if dtype.is_float32 else 700)
        mean = (float(minv) + maxv) / 2
        out = np.random.normal(mean, std_dev, shape).clip(minv, maxv).astype(dtype_str)
    else:
        out = np.random.randint(minv, maxv, shape, dtype_str)

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


def double_elem_width(vdtype, allow_64bit=False):
    if not allow_64bit and vdtype.bits == 32:
        return vdtype

    new_bits = vdtype.bits * 2
    # For the type u8x8, the result type should be u16x8 instead of u16x4.
    new_lanes = vdtype.lanes
    if new_bits * new_lanes > 256:
        new_lanes //= 2
    return vdtype.with_bits(new_bits).with_lanes(new_lanes)


def half_elem_width(vdtype, double_lanes=True):
    assert vdtype.bits >= 8

    # For the type u16x8, maybe the expect result type is u8x8 instead of u8x16.
    new_lanes = vdtype.lanes * 2 if double_lanes else vdtype.lanes
    return vdtype.with_bits(vdtype.bits // 2).with_lanes(new_lanes)


def get_binary_op_result_type(ltype_or_lhs, rtype_or_rhs):
    """Infer the binary operation result type through the same logic of C++ "BinaryOpMatchTypes", in
    addition, adjusting integer literal type is considered."""
    ltype_or_lhs = DataType(ltype_or_lhs) if isinstance(ltype_or_lhs, str) else ltype_or_lhs
    rtype_or_rhs = DataType(rtype_or_rhs) if isinstance(rtype_or_rhs, str) else rtype_or_rhs
    ltype_or_lhs = DataType("float32") if isinstance(ltype_or_lhs, float) else ltype_or_lhs
    rtype_or_rhs = DataType("float32") if isinstance(rtype_or_rhs, float) else rtype_or_rhs
    ltype_or_lhs = DataType("bool") if isinstance(ltype_or_lhs, bool) else ltype_or_lhs
    rtype_or_rhs = DataType("bool") if isinstance(rtype_or_rhs, bool) else rtype_or_rhs
    assert all(isinstance(x, (DataType, int)) for x in (ltype_or_lhs, rtype_or_rhs))

    if all(isinstance(x, int) for x in (ltype_or_lhs, rtype_or_rhs)):
        return "int32"

    ltype, rtype = ltype_or_lhs, rtype_or_rhs
    # Only handle the situation that all operands are integers. For the situation that all operands
    # are floating, because can't know whether a float literal can be represented by float16 or not,
    # so can't do this. For other situations, "binary_op_match_types" is good enough.
    if isinstance(ltype_or_lhs, int):
        if rtype.is_integer and int_within_range(ltype_or_lhs, rtype):
            return rtype.element_of
        ltype = DataType("int32")

    if isinstance(rtype_or_rhs, int):
        if ltype.is_integer and int_within_range(rtype_or_rhs, ltype):
            return ltype.element_of
        rtype = DataType("int32")

    if ltype == rtype:
        return ltype.element_of

    # Promote to higher bits, e.g., i8 + i16 -> i16 + i16, fp16 + fp32 -> fp32 + fp32.
    if (
        (ltype.is_float and rtype.is_float)
        or (ltype.is_int and rtype.is_int)
        or (ltype.is_uint and rtype.is_uint)
    ):
        return (ltype if ltype.bits > rtype.bits else rtype).element_of

    # Cast int -> float when the other operand is float.
    if ltype.is_float and rtype.is_integer:
        return ltype.element_of
    if ltype.is_integer and rtype.is_float:
        return rtype.element_of

    # Handle mixing signed and unsigned integers.
    assert (ltype.is_int and rtype.is_uint) or (ltype.is_uint and rtype.is_int)

    if ltype.bits > rtype.bits:
        return ltype.element_of
    if ltype.bits < rtype.bits:
        return rtype.element_of

    # The width of signed and unsigned integers is same.
    return (ltype if ltype.is_uint else rtype).element_of
