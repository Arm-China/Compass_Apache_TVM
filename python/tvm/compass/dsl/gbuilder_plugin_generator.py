# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Implement the GBuilder plugin generator of DSL program compilation."""
import os
import textwrap
import numpy as np
from .aiff.descriptor import CtrlDescChain


def _gen_ctrl_desc_str(chain):
    ret = ""
    idx = 0
    for desc in chain:
        for i, elem in enumerate(desc):
            if i == 1:
                ret += f"        desc[{idx}] = "
                ret += f"_get_addr(desc_base, (len(ro) + {idx - 1 + len(desc)}) * 4)\n"
            elif elem != 0:
                ret += f"        desc[{idx}] = 0x{elem:08X}\n"
            idx += 1
    return ret


def _is_cognate_np_arr(x, origin):
    if not isinstance(origin, np.ndarray):
        return False

    if x.base is None:
        return x is origin

    return x.base is origin or x.base is origin.base


def _np_arr_offset(x, origin):
    ret = x.ctypes.data - origin.ctypes.data
    assert ret >= 0
    return ret


def _in_out_addr(x, param_infos, args):
    info, offset = None, None
    for i, arg in enumerate(args):
        if _is_cognate_np_arr(x, arg):
            info = param_infos[i]
            offset = _np_arr_offset(x, arg)
            break

    assert info is not None, "Currently all input and output must be parameter of entry function."

    if info.is_input_tensor:
        return f"_get_addr(sgnode.inputs[{info.tensor_idx}], {offset})"
    if info.is_const_tensor:
        return f'_get_addr(sgnode.constants[f"{{nodes[0].name}}/{info.name}"], {offset})'
    assert info.is_output_tensor
    return f"_get_addr(sgnode.outputs[{info.tensor_idx}], {offset})"


def _gen_param_act_desc_str(chain, param_infos, args):
    ret = ""
    idx = 0
    for desc in chain:
        for x in desc:
            if isinstance(x, np.ndarray):
                ret += f"        desc[{idx}] = {_in_out_addr(x, param_infos, args)}\n"
            idx += 1
    return ret


def gen_gb_plugin(param_infos, args, op_type, code_name, output_path, arg_offset, score=10):
    """Generate the GBuilder plugin according to the given information."""
    plugin_str = textwrap.dedent(
        f"""\
        import numpy as np
        from AIPUBuilder.core import BuilderOpPlugin, register_optype, BuilderParams, Tensor
        from AIPUBuilder.plugin_loader import register_plugin, PluginType


        op_type = register_optype("{op_type}")

        def _get_addr(base, offset_in_byte):
            if offset_in_byte == 0:
                return base

            ret = Tensor(base)
            ret.memory_offset().set_base_offset(base.memory_offset())
            ret.memory_offset().relative_offset = offset_in_byte
            return ret

        @register_plugin(PluginType.Builder, 0)
        class {op_type}Plugin(BuilderOpPlugin):
            def get_graph_pattern(self):
                return ([("useless", op_type)], [])

            def get_score(self):
                return {score}

            def set_target(self, target):
                return True

            def check_params(self, nodes):
                return True

            def setup(self, sgnode, nodes):
                sgnode.attrs["keeping_layout"] = False
                return True

            def generate_code_name(self, sgnode, nodes):
                return "{code_name}"

            def generate_descriptor(self, sgnode, nodes):
                desc_base = sgnode.attrs["descriptorbase"]
                ro = BuilderParams()
        """
    )

    # 1. Generate the body of the function "generate_descriptor".
    for i, param_info in enumerate(param_infos):
        if not param_info.is_descriptor:
            continue

        for desc_chain in args[i]:
            plugin_str += f"        desc = [0] * {desc_chain.count_of_u32}\n"
            if isinstance(desc_chain, CtrlDescChain):
                plugin_str += _gen_ctrl_desc_str(desc_chain)
            else:
                plugin_str += _gen_param_act_desc_str(desc_chain, param_infos, args)

            plugin_str += "        for x in desc:\n"
            plugin_str += "            ro.append(x)\n\n"

    plugin_str += "        return ro\n\n"

    # 2. Generate the body of the function "generate_params".
    plugin_str += "    def generate_params(self, sgnode, nodes):\n"
    plugin_str += '        desc_base = sgnode.attrs["descriptorbase"]\n'
    plugin_str += "        ro = BuilderParams()\n"

    cur_desc_offset = 0
    for i, param_info in enumerate(param_infos):
        offset = arg_offset[i]
        plugin_str += "        "
        if param_info.is_input_tensor:
            plugin_str += (
                f"ro.append(_get_addr(sgnode.inputs[{param_info.tensor_idx}], {offset}))\n"
            )
        elif param_info.is_const_tensor:
            plugin_str += f'ro.append(sgnode.constants[f"{{nodes[0].name}}/{param_info.name}"])\n'
        elif param_info.is_output_tensor:
            plugin_str += (
                f"ro.append(_get_addr(sgnode.outputs[{param_info.tensor_idx}], {offset}))\n"
            )
        elif param_info.is_attr:
            dtype = param_info.dtype
            plugin_str += f'value = np.{dtype}(nodes[0].params["{param_info.name}"])\n'
            plugin_str += f'        ro.append(int(value.view("int{dtype.bits}")))\n'
        elif param_info.is_descriptor:
            plugin_str += f"ro.append(_get_addr(desc_base, {cur_desc_offset}))\n"
            cur_desc_offset += args[i].nbytes
        else:
            assert False, "Unsupported parameter information."

    plugin_str += "        return ro\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    open(output_path, "w", encoding="utf-8").write(plugin_str)
