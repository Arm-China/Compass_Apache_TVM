# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
"""Common AIPU utilities."""
import os
import operator
import functools
import tvm
from .. import autotvm, contrib, rpc, tir


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
    """Connect to the RPC tracker and get a RPC session with the RPC key."""
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
    if isinstance(target, tvm.target.Target):
        return target
    assert isinstance(target, str), f"Unsupported target type: {type(target)}."
    if not target.startswith("aipu"):
        target = "aipu -mcpu=" + target
    return tvm.target.Target(target)
