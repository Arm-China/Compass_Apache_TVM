# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Implement the Compass IR generator of DSL program compilation."""
import uuid
import tvm
from tvm import relay, nd
from tvm.relay.op.contrib.aipu_compass import codegen_plugin_register
from ..relax.codegen import CodeGenAipuCompass


def _get_input_ttypes(param_infos, args):
    input_tensors = [y for x, y in zip(param_infos, args) if x.is_input_tensor]
    return [(x.shape, str(x.dtype)) for x in input_tensors]


def _get_output_ttypes(param_infos, args):
    output_tensors = [y for x, y in zip(param_infos, args) if x.is_output_tensor]
    return [(x.shape, str(x.dtype)) for x in output_tensors]


def _get_consts(param_infos, args):
    ret = {}
    for param_info, arg in zip(param_infos, args):
        if param_info.is_const_tensor:
            ret[param_info.name] = relay.Constant(nd.array(arg))
    return ret


def _get_attr_text(param_infos, args):
    ret = "\n"
    for param_info, arg in zip(param_infos, args):
        if param_info.is_attr:
            ret += f"{param_info.name}={arg}\n"
    return ret


def gen_compass_ir(param_infos, args, op_type, ir_txt_path, ir_bin_path):
    """Generate the Compass IR for the corresponding DSL program."""
    assert len(args) == len(param_infos), "Argument count must equal to parameter's."
    input_ttypes = _get_input_ttypes(param_infos, args)
    output_ttypes = _get_output_ttypes(param_infos, args)
    assert len(output_ttypes) > 0, "func should has at least one output."

    # 1. Register the temporary Relay operator that represents the Compass DSL program.
    def _type_rel(arg_types, attrs):  # pylint: disable=unused-argument
        ret = [relay.TensorType(shape, dtype) for shape, dtype in output_ttypes]
        return ret[0] if len(ret) == 1 else relay.TupleType(ret)

    op_name = f"compass_dsl_{uuid.uuid4().hex}"
    relay.op.op.register(op_name, "Temporary operator just for executing Compass DSL program.")
    relay.op.get(op_name).set_num_inputs(len(input_ttypes))
    relay.op.get(op_name).set_attrs_type_key("DictAttrs")
    relay.op.get(op_name).add_type_rel(op_name, _type_rel)

    # 2. Construct the Relay IRModule using the temporary operator.
    params = []
    for i, (shape, dtype) in enumerate(input_ttypes):
        params.append(relay.var(f"input{i}", relay.TensorType(shape, dtype)))
    out = relay.Call(relay.op.get(op_name), params)
    if len(output_ttypes) > 1:
        out = relay.Tuple(list(relay.TupleWrapper(out, len(output_ttypes))))
    func = relay.Function(params, out)
    ir_mod = relay.transform.InferType()(tvm.IRModule.from_expr(func))

    # 3. Register the handler of the temporary operator for generating its Compass IR.
    @codegen_plugin_register(op_name)
    def _gen_compass_dsl(call):  # pylint: disable=unused-variable
        return op_type, call.args, _get_consts(param_infos, args), _get_attr_text(param_infos, args)

    # 4. Get the Compass IR from Relay IR and write them to disk.
    CodeGenAipuCompass().gen2file(ir_mod["main"], ir_txt_path, ir_bin_path)
