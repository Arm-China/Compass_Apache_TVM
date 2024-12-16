# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Implement execute relevant APIs of Zhouyi Compass extension of TIR."""
import os
import copy
import ctypes
import shutil
import textwrap
from subprocess import run
import numpy as np
from tvm import nd, DataType, target as tgt
from ..._ffi.base import _LIB, check_call
from ..logger import WARN
from ..utils import sync_compass_output_dir, check_call_aipu_tool, control_option, get_rpc_session
from .analysis.extract_prim_func_info import ParamInfo
from .aiff.descriptor import DescChainArray, ParamDescChain, ActDescChain
from .compass_ir_generator import gen_compass_ir
from .gbuilder_plugin_generator import gen_gb_plugin


def _unique_merge(np_arrs, name, extra_args):
    for np_arr in np_arrs:
        if all(np_arr is not x for _, x in extra_args):
            extra_args.append((name, np_arr))


def _is_python_program(program):
    """Check whether the specified program is a Python script or not."""
    program_path = shutil.which(program)
    if program_path is None:
        return False

    result = run(["file", program_path], capture_output=True, check=True, text=True)
    if "Python script" in result.stdout:
        return True
    return False


def _get_opencl_debugger_cfg_str(dir_path, num_inputs):
    input_str = "\n"
    for i in range(num_inputs):
        input_str += f"        INPUT_DATA_FILE{i}={dir_path}/input{i}.bin\n"

    return textwrap.dedent(
        f"""\
        #-----------------------------------------------------------------

        GRAPH BINARY FILE
        #-----------------------------------------------------------------
        GRAPH_FILE={dir_path}/aipu.bin
        #-----------------------------------------------------------------

        INPUT DATA CONFIG
        #-----------------------------------------------------------------
        INPUT_DATA_CNT={num_inputs}{input_str}
        #-----------------------------------------------------------------

        #opencl debug on hardware

        #Customers need to configure this parameter themselves

        #SYSTEM_CONFIG_FILE=path of jtag_cfg

        #-----------------------------------------------------------------
        """
    )


class Executor:
    """The class responsible for executing the DSL program.

    Examples
    --------
    .. code-block:: python

        @S.prim_func
        def add_func(a: S.ptr("i8", "global"), b: S.ptr("i8", "global"), n: S.i32):
            xxx

        bm = aipu.tir.BuildManager()
        ex = bm.build(add_func)
    """

    def __init__(self, prim_func_info, output_dir, aipu_info, has_perf_tick, only_sync_diff=False):
        self._func_name = prim_func_info.name
        self._param_infos = prim_func_info.param_infos
        self._output_dir = output_dir
        self._aipu_info = aipu_info
        self._has_perf_tick = has_perf_tick
        self._only_sync_diff = only_sync_diff

        self._gbuilder_dir = f"{self._output_dir}/gbuilder"
        c_code_path = f"{self._gbuilder_dir}/op_lib/{self._func_name}.cl"
        self._c_code = open(c_code_path, encoding="utf-8").read()
        self.mtriple = None
        self._rpc_sess = None
        self._cur_param_infos = None
        self._input_count = None
        self._packed_func_cache = {}
        self._output_nd_arr2np_arr = {}
        self._origin_out_np_arrs = []

    @property
    def c_code(self):
        """Get the Compass OpenCL code of the DSL program.

        Examples
        --------
        .. code-block:: python

            print(ex.c_code)
        """
        return self._c_code

    @property
    def rpc_sess(self):
        """Get or set the RPC session that is used to run a DSL program on a remote device. For
        setting, the value must be an instance of class "tvm.rpc.RPCSession" and already connected
        to the RPC server.

        Examples
        --------
        .. code-block:: python

            from tvm.aipu.utils import get_rpc_session

            ex.rpc_sess = get_rpc_session()
            ex(a, aipu_out, 100)  # Run on remote device through RPC.
        """
        return self._rpc_sess

    @rpc_sess.setter
    def rpc_sess(self, value):
        if not self.mtriple:
            self.mtriple = "aarch64-linux-gnu"
        self._rpc_sess = value

    def _update_for_nullptr(self, args):
        ret = []
        in_nullptr_cnt = 0
        out_nullptr_cnt = 0
        self._cur_param_infos = copy.deepcopy(self._param_infos)

        for param_info, arg in zip(self._cur_param_infos, args):
            if (not param_info.is_attr) and not isinstance(arg, np.ndarray) and arg in (None, 0):
                if param_info.is_input_tensor:
                    in_nullptr_cnt += 1
                elif param_info.is_output_tensor:
                    out_nullptr_cnt += 1
                param_info.category = "attr"
                ret.append(0)
            else:
                if param_info.is_input_tensor:
                    param_info.tensor_idx -= in_nullptr_cnt
                if param_info.is_output_tensor:
                    param_info.tensor_idx -= out_nullptr_cnt
                ret.append(arg)
        return ret

    def _check_param_arg_type(self, args):
        param_cnt, arg_cnt = len(self._cur_param_infos), len(args)
        msg = f'The function "{self._func_name}" expect {param_cnt} args, but got: "{arg_cnt}".'
        assert arg_cnt == param_cnt, msg

        for i, (param_info, arg) in enumerate(zip(self._cur_param_infos, args)):
            if param_info.is_tensor:
                msg = f'The {i + 1}-th arg expect a NumPy, but got: "{type(arg)}".'
                assert isinstance(arg, np.ndarray), msg
                param_elem_dtype = DataType(param_info.dtype).element_of
                if param_elem_dtype != "void" and arg.dtype != param_elem_dtype:
                    msg = f'The {i + 1}-th arg of function "{self._func_name}" expect a '
                    msg += f'{param_elem_dtype} NumPy, but got: "{arg.dtype}".'
                    WARN(msg)

            elif param_info.is_descriptor:
                assert (
                    isinstance(arg, DescChainArray) and len(arg) != 0
                ), f'The {i + 1}-th arg expect a non-empty DescChainArray, but got: "{type(arg)}".'

    def _get_param_info(self, x, args):
        for param_info, arg in zip(self._cur_param_infos, args):
            if x is arg:
                return param_info
        return None  # Indicate it isn't in the arguments.

    def _update_from_descs(self, args):
        # 1. Extract the extra arguments from the descriptors.
        extra_in_args, extra_const_args, extra_out_args = [], [], []
        for param_info, arg in zip(self._cur_param_infos, args):
            if not param_info.is_descriptor:
                continue

            param_name = param_info.name
            for chain in arg:
                if isinstance(chain, ParamDescChain):
                    _unique_merge(chain.const_np_arrs, f"{param_name}_const", extra_const_args)
                elif isinstance(chain, ActDescChain):
                    _unique_merge(chain.in_np_arrs, f"{param_name}_in", extra_in_args)
                    _unique_merge(chain.out_np_arrs, f"{param_name}_out", extra_out_args)

        # 2. Filter out the ones which already appear in the arguments that user provided, at the
        #    same time, check and revise the corresponding "param_info".
        for name_and_x in extra_in_args[:]:
            if self._get_param_info(name_and_x[1], args) is not None:
                extra_in_args.remove(name_and_x)

        for name_and_x in extra_const_args[:]:
            param_info = self._get_param_info(name_and_x[1], args)
            if param_info is not None:
                msg = f'The scope of the parameter "{param_info.name}" expect "global.1".'
                assert param_info.is_const_tensor, msg
                extra_const_args.remove(name_and_x)

        for name_and_x in extra_out_args[:]:
            param_info = self._get_param_info(name_and_x[1], args)
            if param_info is not None:
                param_info.category = "output_tensor"
                extra_out_args.remove(name_and_x)

        # 3. Add the extra arguments and their corresponding "param_info".
        in_idx = len([x for x in self._cur_param_infos if x.is_input_tensor])
        out_idx = len([x for x in self._cur_param_infos if x.is_output_tensor])
        ret = list(args)

        for i, (name, x) in enumerate(extra_in_args):
            ret.append(x)
            param_info = ParamInfo(f"{name}{i}", str(x.dtype), "input_tensor", tensor_idx=in_idx)
            self._cur_param_infos.append(param_info)
            in_idx += 1

        for i, (name, x) in enumerate(extra_const_args):
            ret.append(x)
            self._cur_param_infos.append(ParamInfo(f"{name}{i}", str(x.dtype), "const_tensor"))

        for i, (name, x) in enumerate(extra_out_args):
            ret.append(x)
            param_info = ParamInfo(f"{name}{i}", str(x.dtype), "output_tensor", tensor_idx=out_idx)
            self._cur_param_infos.append(param_info)
            out_idx += 1

        return ret

    def _get_nd_arrs(self, args):
        # The Compass driver only need the tensors, all other arguments will be passed to the DSL
        # program by GBuilder plugin.
        in_np_arrs = []
        out_np_arrs = []
        for param_info, arg in zip(self._cur_param_infos, args):
            if param_info.is_input_tensor:
                in_np_arrs.append(arg)
            elif param_info.is_output_tensor:
                if self._only_sync_diff:
                    self._origin_out_np_arrs.append(arg)
                    arg = arg.copy()
                out_np_arrs.append(arg)

        device = nd.cpu(0) if self.rpc_sess is None else self.rpc_sess.cpu(0)
        in_nd_arrs = tuple(nd.array(x, device) for x in in_np_arrs)
        out_nd_arrs = tuple(nd.array(x, device) for x in out_np_arrs)

        # Record the output arguments for updating after executing.
        self._output_nd_arr2np_arr = dict(zip(out_nd_arrs, out_np_arrs))
        self._input_count = len(in_nd_arrs)
        return in_nd_arrs, out_nd_arrs

    def _build(self, args):
        from tvm.relay.backend.contrib import (  # pylint: disable=import-outside-toplevel
            aipu_compass,
        )

        # 1. Generate the Compass IR.
        op_type = f"DSL_{self._func_name}"
        ir_txt_path = f"{self._gbuilder_dir}/compass_ir.txt"
        ir_bin_path = f"{self._gbuilder_dir}/compass_ir.bin"
        gen_compass_ir(self._cur_param_infos, args, op_type, ir_txt_path, ir_bin_path)

        # 2. Generate the GBuilder plugin.
        plugin_path = f"{self._gbuilder_dir}/aipubt_gb_{self._func_name}.py"
        code_name = f"op_lib/{self._func_name}.o"
        gen_gb_plugin(self._cur_param_infos, args, op_type, code_name, plugin_path)

        # 3. Get the AIPU executable(i.e., aipu.bin) through "aipugb".
        msg = 'The Python version "aipugb" can not be found, please set correct environment!'
        assert _is_python_program("aipugb"), msg

        cmd = ("aipugb", os.path.basename(ir_txt_path), "-w", os.path.basename(ir_bin_path))
        cmd += ("--target", self._aipu_info.name)
        if control_option.is_profile:
            cmd += ("-p", "--enable_dma_perf" if self._has_perf_tick else "--fast_perf")

        check_call_aipu_tool(cmd, self._gbuilder_dir)
        aipu_bin = nd.array(np.fromfile(f"{self._gbuilder_dir}/aipu.bin", dtype="uint8"))

        # 4. Dump files for emulator if needed.
        if control_option.is_emu:
            cmd = ("aipudumper", "aipu.bin", "-i")
            cmd += (",".join(f"input{i}.bin" for i in range(self._input_count)),)
            if control_option.is_profile:
                cmd += ("-p",)
            check_call_aipu_tool(cmd, self._gbuilder_dir)

        # 5. Create the runtime module for subsequent execution.
        # Update the "output_dir" of the singleton and set the function name to empty string, so
        # that the files generated during runtime can be placed to the right directory.
        aipu_compass.AipuCompassBasicConfig.get().common["output_dir"] = self._output_dir
        rt_mod = aipu_compass._ffi_api.AipuCompassModuleNode(aipu_bin, "", self._aipu_info.name, "")
        # The Compass runtime module must to be wrapped by a LLVM module, otherwise it can't be
        # exported and used through TVM RPC.
        llvm_target_str = f"llvm -mtriple={self.mtriple}" if self.mtriple else "llvm"
        return tgt._ffi_api.AttachCompassModuleToLLVM(rt_mod, llvm_target_str)

    def _get_packed_funcs(self, args):
        # Get the real build result, the cache is handled by "_build".
        local_rt_mod = self._build(args)

        key = (local_rt_mod, self.rpc_sess)
        if key in self._packed_func_cache:
            return self._packed_func_cache[key]

        rt_mod = local_rt_mod
        if self.rpc_sess:
            export_path = f"{self._output_dir}/{self._func_name}_{id(local_rt_mod)}.so"
            local_rt_mod.export_library(export_path, cc=os.getenv("AIPU_TVM_DEVICE_COMPILER"))
            self.rpc_sess.upload(export_path)
            rt_mod = self.rpc_sess.load_module(os.path.basename(export_path))

        # Get and cache the executable packed function.
        compass_set_inputs = rt_mod.get_function("compass_set_inputs", query_imports=True)
        compass_set_outputs = rt_mod.get_function("compass_set_outputs", query_imports=True)
        compass_get_outputs = rt_mod.get_function("compass_get_outputs", query_imports=True)
        compass_run = rt_mod.get_function("compass_run", query_imports=True)

        ret = (compass_set_inputs, compass_set_outputs, compass_get_outputs, compass_run, rt_mod)
        self._packed_func_cache[key] = ret
        return ret

    def _sync_outputs(self):
        # Copy the real output data to the object that user provide, user provide output argument as
        # a numpy array, but the real output data is a local or remote NDArray.
        if self._only_sync_diff:
            for i, (nd_arr, np_arr) in enumerate(self._output_nd_arr2np_arr.items()):
                new_np_arr = nd_arr.numpy()
                diff_indices = new_np_arr != np_arr
                self._origin_out_np_arrs[i][diff_indices] = new_np_arr[diff_indices]
            return

        for nd_arr, np_arr in self._output_nd_arr2np_arr.items():
            assert np_arr.flags["C_CONTIGUOUS"]
            data = np_arr.ctypes.data_as(ctypes.c_void_p)
            nbytes = ctypes.c_size_t(np_arr.size * np_arr.dtype.itemsize)
            check_call(_LIB.TVMArrayCopyToBytes(nd_arr.handle, data, nbytes))

    def _run(self, args):
        # 1. Update and check arguments.
        args = self._update_for_nullptr(args)
        self._check_param_arg_type(args)
        args = self._update_from_descs(args)

        # 2. Dump input tensors if needed.
        in_nd_arrs, out_nd_arrs = self._get_nd_arrs(args)
        if control_option.is_debug or control_option.is_emu:
            for i, nd_arr in enumerate(in_nd_arrs):
                nd_arr.numpy().tofile(f"{self._gbuilder_dir}/input{i}.bin")

        if control_option.is_debug:
            # Generate the configuration file used by OpenCL Debugger.
            cfg_str = _get_opencl_debugger_cfg_str(self._gbuilder_dir, len(in_nd_arrs))
            open(f"{self._gbuilder_dir}/opencl_debug.cfg", "w", encoding="utf-8").write(cfg_str)

        # 3. Get the executable packed function, the RPC and cache is handled by us.
        _, compass_set_outputs, _, compass_run, _ = self._get_packed_funcs(args)

        # 4. Run the compiled DSL program.
        compass_set_outputs(*out_nd_arrs)  # Initialize the output memory space.
        compass_run(*(in_nd_arrs + out_nd_arrs))

        # 5. Dump output tensors if needed.
        if control_option.is_debug or control_option.is_emu:
            for i, nd_arr in enumerate(out_nd_arrs):
                nd_arr.numpy().tofile(f'{self._gbuilder_dir}/output.bin{"" if i == 0 else i}')

        # 6. Sync the output to the corresponding NumPy object provided by user.
        self._sync_outputs()

    def _post_profile(self):
        # Download the profile data from the remote device.
        sync_compass_output_dir(self.rpc_sess, lambda x: x.endswith("/profile_data.bin"))

        # Generate the report through profiler.
        runtime_dir = f"{self._output_dir}/runtime"
        output_path = f"{runtime_dir}/profile_output.html"
        cmd = ("aipu_profiler", f"{self._gbuilder_dir}/graph.json")
        cmd += (f"{runtime_dir}/profile_data.bin", "-o", output_path)
        check_call_aipu_tool(cmd, runtime_dir)

        # Print out the most useful information directly.
        total_cycles = int(open(output_path, "r").read().split('"Total Cycles": ')[1].split(",")[0])
        print(f"Total cycles from profiler: {total_cycles}")
        print(f'For more details about the profiler report, please see "{output_path}".')
        return total_cycles

    def run(self, *args):
        """The run end-to-end interface of DSL program execution.

        Parameters
        ----------
        args : List
            The execution arguments, which should be aligned with the parameters of the DSL program.

        Examples
        --------
        .. code-block:: python

            from tvm.aipu.utils import rand

            a = rand(100, "int8")
            aipu_out = np.empty(100, dtype="int8")
            ex.run(a, aipu_out, 100)  # Can also be "ex(a, aipu_out, 100)".
        """
        if control_option.is_rpc:  # Overwrite according to environment variable settings.
            self.rpc_sess = get_rpc_session(rpc_key=control_option.rpc_key)

        self._run(args)

        if self.rpc_sess is not None and control_option.is_profile:
            self._post_profile()

    def __call__(self, *args):
        self.run(*args)

    def benchmark(
        self,
        *args,
        repeat=2,
        number=3,
        min_repeat_ms=0,
        limit_zero_time_iterations=100,
        cooldown_interval_ms=0,
        repeats_to_cooldown=1,
    ):
        """Calculate runtime of a function by repeatedly calling it.

        Use this function to get an accurate measurement of the runtime of a function. The function
        is run multiple times in order to account for variability in measurements, processor speed
        or other external factors.  Mean, median, standard deviation, min and max runtime are all
        reported. On the AIPU specifically, synchonization and data transfer operations are not
        counted towards the runtime. This allows for fair comparison of runtimes across different
        functions and models.

        The benchmarking loop looks approximately like so:

        .. code-block:: python

            for r in range(repeat):
                time_start = now()
                for n in range(number):
                    func_name()
                time_end = now()
                total_times.append((time_end - time_start)/number)


        Parameters
        ----------
        args: Sequence[Object]
            Arguments to the function. These are cached before running timing code, so that data
            transfer costs are not counted in the runtime.

        repeat: Optional[int]
            Number of times to run the outer loop of the timing code (see above). The output will
            contain `repeat` number of datapoints.

        number: Optional[int]
            Number of times to run the inner loop of the timing code. This inner loop is run in
            between the timer starting and stopping. In order to amortize any timing overhead,
            `number` should be increased when the runtime of the function is small (less than a 1/10
            of a millisecond).

        min_repeat_ms: Optional[int]
            If set, the inner loop will be run until it takes longer than `min_repeat_ms`
            milliseconds. This can be used to ensure that the function is run enough to get an
            accurate measurement.

        limit_zero_time_iterations: Optional[int]
            The maximum number of repeats when measured time is equal to 0.
            It helps to avoid hanging during measurements.

        cooldown_interval_ms: Optional[int]
            The cooldown interval in milliseconds between the number of repeats defined by
            `repeats_to_cooldown`.

        repeats_to_cooldown: Optional[int]
            The number of repeats before the cooldown is activated.

        Note
        ----
        The function will be invoked  (1 + number x repeat) times,
        with the first call discarded in case there is lazy initialization.

        Returns
        -------
        ret: BenchmarkResult
            Runtimes of the function. Use `.mean` to access the mean runtime, use `.results` to
            access the individual runtimes (in seconds).

        Examples
        --------
        .. code-block:: python

            print(ex.benchmark(a, aipu_out, n))

        See Also
        --------
        - :doc:`../getting_started/tutorials/0_quick_start`
        """
        args = self._update_for_nullptr(args)
        self._check_param_arg_type(args)
        args = self._update_from_descs(args)

        in_nd_arrs, out_nd_arrs = self._get_nd_arrs(args)

        # Get the executable packed function, the RPC and cache is handled by us.
        tmp = self._get_packed_funcs(args)
        compass_set_inputs, compass_set_outputs, compass_get_outputs, _, rt_mod = tmp

        compass_set_inputs(*in_nd_arrs)
        compass_set_outputs(*out_nd_arrs)  # Initialize the output memory space.

        # Create timing function and run.
        ret = rt_mod.time_evaluator(
            func_name="compass_execute",
            dev=(nd.cpu(0) if self.rpc_sess is None else self.rpc_sess.cpu(0)),
            number=number,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            limit_zero_time_iterations=limit_zero_time_iterations,
            cooldown_interval_ms=cooldown_interval_ms,
            repeats_to_cooldown=repeats_to_cooldown,
        )()

        compass_get_outputs(*out_nd_arrs)
        self._sync_outputs()
        return ret

    def profile(self, *args):
        """Collect accurate performance information on remote device and return total cycles.

        Parameters
        ----------
        args : List
            The execution arguments, which should be aligned with the parameters of the DSL program.

        Returns
        -------
        total_cycles : int
            The total hardware cycles it took to execute the DSL program.

        Examples
        --------
        .. code-block:: python

            ex.profile(a, aipu_out, 100)

        See Also
        --------
        - :doc:`../../how_to_guides/how_to_use_profiler`

        """
        assert self.rpc_sess is not None, "Please set the RPC session first."
        msg = "Can't support dump emulator files without profile option from this API."
        assert not (control_option.is_emu and not control_option.is_profile), msg

        old_value, control_option.is_profile = control_option.is_profile, True
        self._run(args)
        control_option.is_profile = old_value

        return self._post_profile()
