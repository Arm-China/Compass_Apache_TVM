# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Common Compass utilities."""
import os
import re
import textwrap
from subprocess import run, STDOUT
from .. import rpc, target as tgt
from .logger import INFO
from .runtime import CompassBasicConfig


_EXE_NAME2TOOL_NAME = {
    "aipuopt": "Optimizer",
    "aipugb": "GBuilder",
    "aipugsim": "GSim",
    "aipurun": "AIPURun",
    "aipu_profiler": "Profiler",
    "aipudumper": "Dumper",
}


def check_call_compass_tool(cmd, work_dir=os.getcwd()):
    """Call tools of AIPUBuilder through sub process and check the return code."""
    work_dir = os.path.abspath(work_dir)
    if work_dir != os.getcwd():
        os.makedirs(work_dir, exist_ok=True)

    exe_name = cmd[0]
    log_file = f"{work_dir}/{exe_name}.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Current Working Directory: {work_dir}\nCommand Line: {' '.join(cmd)}\n")
        f.write("Standard Output & Error:\n")
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
            cwd=work_dir,
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

    if ret_code != 0 or count_errors != 0:
        raise RuntimeError(
            f"Error happened when executing the Compass {_EXE_NAME2TOOL_NAME[exe_name]}, for more "
            f'details, please refer to the log file "{log_file}".'
        )


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
        If rpc_key = "None", get it from env "CPS_TVM_RPC_KEY".

    tracker_host : Optional[str]
        The hostname or IP address of the RPC tracker.
        If tracker_host = "None", get it from env "CPS_TVM_RPC_TRACKER_IP".

    tracker_port: Optional[int, str]
        The port of the RPC tracker.
        If tracker_port = "None", get it from env "CPS_TVM_RPC_TRACKER_PORT".

    priority : Optional[int]
        The priority of the request.
        If priority = "None", get it from env "CPS_TVM_RPC_PRIORITY".

    Returns
    -------
    sess : tvm.rpc.RPCSession
        The RPC session that is already connected to the RPC server.
    """
    # Override logic of RPC key is special, function argument has higher priority.
    rpc_key = rpc_key or os.getenv("CPS_TVM_RPC_KEY")
    assert rpc_key, 'Set RPC key through arg or env "CPS_TVM_RPC_KEY", "CPS_DSL".'

    tracker_host = os.getenv("CPS_TVM_RPC_TRACKER_IP") or tracker_host
    assert tracker_host, 'Set RPC tracker host through arg or env "CPS_TVM_RPC_TRACKER_IP".'
    tracker_port = os.getenv("CPS_TVM_RPC_TRACKER_PORT") or tracker_port
    assert tracker_port, 'Set RPC tracker port through arg or env "CPS_TVM_RPC_TRACKER_PORT".'
    priority = os.getenv("CPS_TVM_RPC_PRIORITY") or priority
    assert priority, 'Set RPC priority through arg or env "CPS_TVM_RPC_PRIORITY".'

    valid_rpc_keys = os.getenv("CPS_TVM_VALID_RPC_KEYS")
    if valid_rpc_keys:
        valid_rpc_keys = tuple(x.strip() for x in valid_rpc_keys.split("|") if x.strip() != "")
        assert (
            rpc_key in valid_rpc_keys
        ), f"Invalid RPC key '{rpc_key}', the valid choices are {valid_rpc_keys}."

    return rpc.connect_tracker(tracker_host, int(tracker_port)).request(
        key=rpc_key, priority=int(priority), session_timeout=session_timeout
    )


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
    msg = f'The arg "rpc_sess" expect a RPC session, but got: "{type(rpc_sess)}".'
    assert isinstance(rpc_sess, rpc.RPCSession), msg

    remote_files = tuple(
        x for x in rpc_sess.list_files(".") if x.startswith("compass_output") and filter_fn(x)
    )
    local_output_dir = CompassBasicConfig.get().common["output_dir"]

    for remote_file in remote_files:
        local_full_path = f"{local_output_dir}/{remote_file.split(os.path.sep, 1)[1]}"
        os.makedirs(os.path.dirname(local_full_path), exist_ok=True)
        open(local_full_path, "wb").write(rpc_sess.download(remote_file))
        INFO(f'Downloaded remote "{remote_file}" as local "{local_full_path}".')


class _ControlOption:
    _acceptable_names = {
        "debug": ("d", "dbg", "debug"),
        "profile": ("p", "prf", "profile", "profiler", "profilor"),
        "emu": ("e", "emu", "emulate", "emulator", "emulater"),
        "random_pause": ("rp", "random_pause", "random-pause", "randompause"),
        "asm": ("a", "asm", "assembly"),
        "random_seed": ("rs", "random_seed", "random-seed", "randomseed"),
        "rpc": ("r", "rpc"),
    }
    _help_msg = textwrap.dedent(
        f'''
        Supported Options:
            {", ".join(_acceptable_names["debug"])}
                Add "-O0 -g" to OpenCL compiler, generate OpenCL debugger config file, dump input
                and output files.
            {", ".join(_acceptable_names["emu"])}
                Dump all needed files for running on emulator.
            {", ".join(_acceptable_names["profile"])}
                Add profile relevant options to "aipugb", "aipudumper", and generate the report
                automatically when running on remote device through RPC.
            {", ".join(_acceptable_names["random_pause"])}
                Pause each TEC thread a random length of time after each synchronization point when
                running PySim, used to reproduce and debug the multiple TEC synchronization bug.
            {", ".join(_acceptable_names["asm"])}
                Dump the Compass assembly code during compiling the generated Compass OpenCL code.
            {", ".join(f"{x}=<seed>" for x in _acceptable_names["random_seed"])}
                Set the random seed used by function "rand", so the flaky test cases can be
                reproduced easily.
            {", ".join(f"{x}[=<rpc_key>]" for x in _acceptable_names["rpc"])}
                Run the DSL program on a remote device through RPC, if the RPC key isn't given, the
                value of environment variable "CPS_TVM_RPC_KEY" will be used.
        Example:
            setenv CPS_DSL "e;p"
            setenv CPS_DSL "d;rs=1195868741"
            setenv CPS_DSL "r=juno,X2_1204;p"'''
    )

    def __init__(self):
        self.is_debug = False
        self.is_emu = False
        self.is_profile = False
        self.is_random_pause = False
        self.is_asm = False
        self.random_seed = None
        self.random_seed_is_set = False
        self.is_rpc = False
        self.rpc_key = None

        options = set(x.strip() for x in os.getenv("CPS_DSL", "").split(";")) - {""}
        normalized_options = []
        for option in options:
            msg = f'Unsupported option "{option}" in environment "CPS_DSL"\n{self._help_msg}'

            name, *value = option.split("=", maxsplit=1)
            value = None if len(value) == 0 else value[0]

            if name in self._acceptable_names["debug"]:
                self.is_debug = True
                normalized_options.append("debug")
            elif name in self._acceptable_names["emu"]:
                self.is_emu = True
                normalized_options.append("emu")
            elif name in self._acceptable_names["profile"]:
                self.is_profile = True
                normalized_options.append("profile")
            elif name in self._acceptable_names["random_pause"]:
                self.is_random_pause = True
                normalized_options.append("random_pause")
            elif name in self._acceptable_names["asm"]:
                self.is_asm = True
                normalized_options.append("asm")
            elif name in self._acceptable_names["random_seed"]:
                assert value is not None and value.isdigit(), msg
                self.random_seed = int(value)
                normalized_options.append(f"random_seed={self.random_seed}")
            elif name in self._acceptable_names["rpc"]:
                self.is_rpc = True
                self.rpc_key = os.getenv("CPS_TVM_RPC_KEY") if value is None else value
                normalized_options.append("rpc" if value is None else f"rpc={self.rpc_key}")
            else:
                raise ValueError(msg)

        if len(normalized_options) != 0:
            INFO(f'CPS_DSL is set to "{"; ".join(normalized_options)}"')


control_option = _ControlOption()


def canonicalize_target(target):
    """Canonicalize target and return tvm.target.Target."""
    if "CPS_TVM_GBUILDER_TARGET" in os.environ:
        target = os.environ["CPS_TVM_GBUILDER_TARGET"]
    elif control_option.rpc_key is not None:
        target = control_option.rpc_key.split(",")[1]

    if isinstance(target, tgt.Target):
        return target
    assert isinstance(target, str), f"Unsupported target type: {type(target)}."
    if not target.startswith("compass"):
        target = "compass -mcpu=" + target
    return tgt.Target(target)
