# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Implement the simple run through Compass simulator."""
import os
import textwrap
from ... import DataType
from ..utils import control_option
from .analysis.extract_prim_func_info import PrimFuncInfo, ParamInfo
from .cc import compile_c
from .executor import Executor


_DTYPE_NP2CL = {
    "int8": "char",
    "int16": "short",
    "int32": "int",
    "uint8": "uchar",
    "uint16": "ushort",
    "uint32": "uint",
    "float16": "half",
    "float32": "float",
    "bfloat16": "__bf16",
}


def _gen_c_code(inputs, consts, outputs, descs, code_snippet):
    params = []
    for name, x in inputs + consts + outputs:
        params.append(f"__global {_DTYPE_NP2CL[str(x.dtype)]}* {name}")

    for name, _ in descs:
        params.append(f"__global uint* {name}")

    return textwrap.dedent(
        f"""\
        #include <compass/dsl.h>

        __kernel void run_sim({", ".join(params)}) {{
          if (get_local_id(0) != 0) {{
            barrier(CLK_LOCAL_MEM_FENCE);return;
          }}

          {code_snippet}

          barrier(CLK_LOCAL_MEM_FENCE);
        }}
        """
    )


def _gen_prim_func_info(inputs, consts, outputs, descs):
    ret = PrimFuncInfo("run_sim", len(inputs) + len(consts) + len(outputs) + len(descs))
    idx = 0

    for i, (name, x) in enumerate(inputs):
        ret.param_infos[idx] = ParamInfo(name, DataType(x.dtype), "input_tensor", tensor_idx=i)
        idx += 1

    for name, x in consts:
        ret.param_infos[idx] = ParamInfo(name, DataType(x.dtype), "const_tensor")
        idx += 1

    for i, (name, x) in enumerate(outputs):
        ret.param_infos[idx] = ParamInfo(name, DataType(x.dtype), "output_tensor", tensor_idx=i)
        idx += 1

    for name, _ in descs:
        ret.param_infos[idx] = ParamInfo(name, DataType("uint32"), "descriptor")
        idx += 1

    return ret


def run_sim(cps_info, out_dir, code_snippet, inputs=None, outputs=None, consts=None, descs=None):
    """Simple run the specified code snippet through Compass simulator."""
    inputs = tuple() if inputs is None else inputs
    outputs = tuple() if outputs is None else outputs
    consts = tuple() if consts is None else consts
    descs = tuple() if descs is None else descs

    c_code = _gen_c_code(inputs, consts, outputs, descs, code_snippet)
    out_path = f"{out_dir}/gbuilder/op_lib/run_sim.o"
    compile_c(c_code, cps_info.name, "", out_path, os.path.dirname(out_path))

    prim_func_info = _gen_prim_func_info(inputs, consts, outputs, descs)
    executor = Executor(prim_func_info, out_dir, cps_info, False, only_sync_diff=True)

    args = [x for _, x in inputs] + [x for _, x in consts] + [x for _, x in outputs]
    args += [x for _, x in descs]
    old_value, control_option.is_rpc = control_option.is_rpc, False
    executor.run(*args)
    control_option.is_rpc = old_value
