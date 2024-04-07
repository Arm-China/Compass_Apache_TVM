# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=too-many-nested-blocks
"""Build all the AIPU subgraphs."""
import re
import json
import tvm
from tvm import relay, ir, tir
from tvm.relay.op.contrib import aipu_compass as aipu_rly_op
from ..config import AipuCompassFunctionConfig
from ..engine import create_forward_engine


class QuantInfo:
    """Quantization information for each input and output node."""

    def __init__(self, dtype, scale, zero_point):
        self.dtype = dtype
        self.scale = scale
        self.zero_point = zero_point


def _create_quant_info_list(layers, param_names):
    ret = [None] * len(param_names)
    for layer in layers:
        for line in layer.split("\n"):
            if len(line) == 0:
                continue
            key, value = (x.strip() for x in line.split("="))
            if key == "layer_top":
                names = [x.strip() for x in value.strip("[]").split(",")]
            elif key == "layer_top_type":
                dtypes = [x.strip() for x in value.strip("[]").split(",")]
            elif key == "layer_top_scale":
                scales = json.loads(value)
            elif key == "layer_top_zp":
                zero_points = json.loads(value)

        for i, name in enumerate(names):
            if name not in param_names:
                continue
            # The scale and zero point used by AIPU Compass have small difference with those of TVM.
            ret[param_names.index(name)] = QuantInfo(dtypes[i], (1 / scales[i]), -zero_points[i])
    return ret


def _parse_in_out_quant_info(quant_compass_ir_txt_path):
    q_cir = open(quant_compass_ir_txt_path).read()
    # Get the name list of input and output nodes.
    matches = re.search(r"input_tensors\s*=\s*\[(.+)\]", q_cir, re.MULTILINE)
    assert matches and len(matches.groups()) == 1
    in_param_names = [x.strip() for x in matches.group(1).split(",")]
    matches = re.search(r"output_tensors\s*=\s*\[(.+)\]", q_cir, re.MULTILINE)
    assert matches and len(matches.groups()) == 1
    out_param_names = [x.strip() for x in matches.group(1).split(",")]

    # Get the layer list of producing input and output nodes.
    in_param_layers = []
    out_param_layers = []
    for layer in q_cir.split("layer_type"):
        prefix = r"layer_top\s*=\s*\[.*("
        suffix = r").*\]"
        if re.search(f"{prefix}{'|'.join(in_param_names)}{suffix}", layer, re.MULTILINE):
            in_param_layers.append(layer)
        if re.search(f"{prefix}{'|'.join(out_param_names)}{suffix}", layer, re.MULTILINE):
            out_param_layers.append(layer)

    # Create object to store quantization information of each input and output nodes.
    return (
        _create_quant_info_list(in_param_layers, in_param_names),
        _create_quant_info_list(out_param_layers, out_param_names),
    )


class AipuSubgraphBuilder:
    """Build the given AIPU subgraph and store the result in function attribute."""

    def __init__(self, forward_engine_type):
        self._forward_engine_type = forward_engine_type
        self._forward_engine = create_forward_engine(forward_engine_type)

    def build(self, func):
        """Build the given AIPU subgraph and store the result in function attribute."""
        # 1. Process the AIPU subgraph through the specific forward engine.
        new_func = self._forward_engine.pre_build(func)
        new_func = new_func.with_attr("compass.pre_build", self._forward_engine_type)

        # 2. Get quantization information of input and output nodes.
        in_quant_info, out_quant_info = None, None
        # Only "opt_float" don't need to inject nodes and change the input and output type of the
        # AIPU subgraph, otherwise obtain them from quantized Compass IR.
        if self._forward_engine_type != "opt_float":
            cfg = AipuCompassFunctionConfig(new_func.attrs.global_symbol)
            in_quant_info, out_quant_info = _parse_in_out_quant_info(cfg.quant_compass_ir_path[0])

        # 3. Change the type of each input node if needed.
        new_params = new_func.params
        if in_quant_info:
            new_params = []
            assert len(in_quant_info) == len(func.params)
            for old_param, quant_info in zip(func.params, in_quant_info):
                new_type = ir.TensorType(old_param.checked_type.shape, quant_info.dtype)
                new_params.append(relay.var(old_param.name_hint, new_type))

        # 4. Change the type of each output node if needed.
        new_ret_type = old_ret_type = func.ret_type
        if out_quant_info:
            is_multiple_output = isinstance(old_ret_type, ir.TupleType)
            old_types = old_ret_type.fields if is_multiple_output else (old_ret_type,)

            new_types = []
            assert len(out_quant_info) == len(old_types)
            for old_type, quant_info in zip(old_types, out_quant_info):
                new_types.append(ir.TensorType(old_type.shape, quant_info.dtype))

            new_ret_type = ir.TupleType(new_types) if is_multiple_output else new_types[0]

        # 5. Shrink the original AIPU subgraph to one node and as the body of the new function.
        new_body = aipu_rly_op.custom_op(new_params, new_ret_type)
        new_func = relay.Function(new_params, new_body, attrs=new_func.attrs)
        return new_func, in_quant_info, out_quant_info


class CallerOpInjector(relay.ExprMutator):
    """Inject necessary quantization or type conversion nodes to the caller."""

    def __init__(self, quant_info, bypass_quant_info=()):
        super().__init__()
        self._quant_info = quant_info
        self._bypass_quant = bypass_quant_info[0]
        self._bypass_dequant = bypass_quant_info[1]

    def visit_var(self, var):
        ret = super().visit_var(var)
        if var in self._bypass_quant:
            return self._bypass_quant[var]
        return ret

    def visit_call(self, call):
        ret = super().visit_call(call)

        if ret.op not in self._quant_info:
            return ret

        # This node is a call to a changed AIPU subgraph.
        in_quant_info, out_quant_info = self._quant_info[ret.op]
        # Inject necessary nodes after the corresponding arguments.
        new_args = []
        assert len(in_quant_info) == len(ret.args)
        for i, (old_arg, quant_info) in enumerate(zip(ret.args, in_quant_info)):
            old_dtype = tvm.DataType(call.args[i].checked_type.dtype)
            quant_dtype = tvm.DataType(quant_info.dtype)

            if call.args[i] in self._bypass_quant:
                new_args.append(old_arg)
                continue

            # Maybe need insert type conversion node for the int or uint parameter.
            if old_dtype.type_code in (tvm.DataTypeCode.INT, tvm.DataTypeCode.UINT):
                if old_dtype == quant_dtype:
                    new_args.append(old_arg)
                else:
                    a_min = float(tir.min_value(quant_dtype).value)
                    a_max = float(tir.max_value(quant_dtype).value)
                    new_args.append(relay.cast(relay.clip(old_arg, a_min, a_max), quant_dtype))
                continue

            # Need insert quantize node for the float parameter.
            assert str(old_dtype) == "float32"
            scale = relay.const(quant_info.scale, "float32")
            zp = relay.const(quant_info.zero_point, "int32")  # pylint: disable=invalid-name
            new_args.append(relay.qnn.op.quantize(old_arg, scale, zp, out_dtype=quant_info.dtype))

        # Create the new call with the new arguments.
        ret = relay.expr.CallWithFields(ret, args=new_args)

        # Inject necessary nodes after the corresponding output nodes.
        old_ret_type = call.checked_type
        is_multiple_output = isinstance(old_ret_type, ir.TupleType)
        old_types = old_ret_type.fields if is_multiple_output else (old_ret_type,)

        out_exprs = []
        assert len(out_quant_info) == len(old_types)
        need_bypass = ret.op in self._bypass_dequant
        for i, quant_info in enumerate(out_quant_info):
            out_expr = relay.TupleGetItem(ret, i) if is_multiple_output else ret
            old_dtype = tvm.DataType(old_types[i].dtype)

            if need_bypass and i in self._bypass_dequant[ret.op]:
                out_exprs.append(out_expr)
                continue

            # Maybe need insert type conversion node for the int or uint output node.
            if old_dtype.type_code in (tvm.DataTypeCode.INT, tvm.DataTypeCode.UINT):
                if old_dtype == tvm.DataType(quant_info.dtype):
                    out_exprs.append(out_expr)
                else:
                    a_min = float(tir.min_value(old_dtype).value)
                    a_max = float(tir.max_value(old_dtype).value)
                    out_exprs.append(relay.cast(relay.clip(out_expr, a_min, a_max), old_dtype))
                continue

            # Need insert dequantize node for the float output node.
            assert str(old_dtype) in ("float16", "float32")
            scale = relay.const(quant_info.scale, "float32")
            zero_point = relay.const(quant_info.zero_point, "int32")
            out_exprs.append(relay.qnn.op.dequantize(out_expr, scale, zero_point, -1, old_dtype))

        return relay.Tuple(out_exprs) if is_multiple_output else out_exprs[0]


class OpCanonicalizer(relay.ExprMutator):
    """Canonicalize some invalid operators."""

    def visit_call(self, call):
        post = super().visit_call(call)

        if post.op == relay.op.get("clip"):
            clip = post
            a_min, a_max = clip.attrs.a_min, clip.attrs.a_max
            dtype = call.checked_type.dtype
            valid_min = float(tir.min_value(dtype).value)
            valid_max = float(tir.max_value(dtype).value)

            # Useless clip, just delete it.
            if a_min <= valid_min and a_max >= valid_max:
                return post.args[0]

            # Normal clip, just keep it.
            if a_min >= valid_min and a_max <= valid_max:
                return post

            # Useful clip but "a_min" or "a_max" need to be adjusted.
            # Don't try "new_attrs = dict(clip.attrs)", will error in below "ir.make_node", because
            # the key must be string.
            new_attrs = {str(k): clip.attrs[k] for k in clip.attrs.keys()}
            new_attrs["a_min"] = max(a_min, valid_min)
            new_attrs["a_max"] = min(a_max, valid_max)
            new_attrs = ir.make_node(str(clip.attrs).split("(")[0], **new_attrs)

            return relay.expr.CallWithFields(post, attrs=new_attrs)

        return post


class RedundantTupleCleaner(relay.ExprMutator):
    """Remove the redundant tuples."""

    def visit_tuple(self, tup):
        post = super().visit_tuple(tup)

        # The type of each item of the tuple must be "TupleGetItem".
        if any(not isinstance(x, relay.TupleGetItem) for x in post.fields):
            return post

        # The tuple value of each "TupleGetItem" must be same.
        src_expr = post.fields[0].tuple_value
        if any(x.tuple_value != src_expr for x in post.fields):
            return post

        # The two tuple must have the same length.
        if len(src_expr.checked_type.fields) != len(post.fields):
            return post

        # The position of each item of the two tuple must be same.
        if any(x.index != i for i, x in enumerate(post.fields)):
            return post

        return src_expr


@ir.transform.module_pass(opt_level=0, required=("InferType",))
class BuildAipuSubgraph:
    """Build the AIPU subgraphs, store the result in the attribute of the corresponding function,
    shrink the body of each AIPU subgraph to one node and inject necessary quantization or type
    conversion nodes for its input and output nodes."""

    def __init__(self, forward_engine, bypass_input_quant=None, bypass_output_dequant=None):
        self._builder = AipuSubgraphBuilder(forward_engine)
        self.bypass_input_quant = bypass_input_quant
        self.bypass_output_dequant = bypass_output_dequant

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        new_ir_mod = ir_mod.shallow_copy()
        # Delete the caller first, so the type inference can work correctly.
        new_ir_mod.remove("main")

        # 1. Build all AIPU subgraphs, and change each AIPU subgraph according to quantization.
        quant_info = dict()
        pattern_items = []
        for gvar, func in ir_mod.functions.items():
            if gvar.name_hint.startswith("tvmgen_default_aipu_compass_main_"):
                pattern_items.append((gvar, func))
        # Sorted by gvar name_hint and self._builder.build in this order.
        sorted_pattern_items = sorted(
            pattern_items, key=lambda item: int(item[0].name_hint.split("_")[-1])
        )
        for gvar, func in sorted_pattern_items:
            if not hasattr(func.attrs, "Compiler") or func.attrs.Compiler != "aipu_compass":
                continue
            new_ir_mod[gvar], in_quant_info, out_quant_info = self._builder.build(func)

            if in_quant_info and out_quant_info:
                quant_info[gvar] = (in_quant_info, out_quant_info)

            # Do type inference once a new function is updated into the IRModule.
            new_ir_mod = relay.transform.InferType()(new_ir_mod)

        # 2. Inject necessary quantization or type conversion nodes in caller.
        new_ir_mod["main"] = ir_mod["main"]
        bypass_dequant_dict = dict()
        bypass_quant_dict = dict()
        if self.bypass_output_dequant:
            expr_outs = [new_ir_mod["main"].body]
            if isinstance(new_ir_mod["main"].ret_type, ir.TupleType):
                expr_outs = new_ir_mod["main"].body.fields
            for out_idx in self.bypass_output_dequant:
                out_expr = expr_outs[out_idx]
                if isinstance(out_expr, relay.Call) and isinstance(out_expr.op, relay.GlobalVar):
                    gfunc = new_ir_mod[out_expr.op]
                    assert not isinstance(gfunc.ret_type, ir.TupleType)
                    if hasattr(gfunc.attrs, "Compiler") and gfunc.attrs.Compiler == "aipu_compass":
                        name = out_expr.op
                        if name not in bypass_dequant_dict:
                            bypass_dequant_dict[name] = []
                        bypass_dequant_dict[name].append(0)
                elif isinstance(out_expr, relay.TupleGetItem):
                    idx = int(out_expr.index)
                    tuple_value = out_expr.tuple_value
                    if isinstance(tuple_value, relay.Call) and isinstance(
                        tuple_value.op, relay.GlobalVar
                    ):
                        gfunc = new_ir_mod[tuple_value.op]
                        if (
                            hasattr(gfunc.attrs, "Compiler")
                            and gfunc.attrs.Compiler == "aipu_compass"
                        ):
                            name = tuple_value.op
                            if name not in bypass_dequant_dict:
                                bypass_dequant_dict[name] = []
                            bypass_dequant_dict[name].append(idx)

        if self.bypass_input_quant:
            expr_ins = [new_ir_mod["main"].params[idx] for idx in self.bypass_input_quant]
            for inp in expr_ins:
                consumers = relay.analysis.consumers(inp, new_ir_mod["main"].body)

                def _compass_consumers(call):
                    if isinstance(call, relay.Call) and isinstance(call.op, relay.GlobalVar):
                        gfunc = new_ir_mod[call.op]
                        if (
                            hasattr(gfunc.attrs, "Compiler")
                            and gfunc.attrs.Compiler == "aipu_compass"
                        ):
                            return True
                    return False

                if all([_compass_consumers(call) for call in consumers]):
                    new_dtype = None
                    for call in consumers:
                        checked_type = None
                        for idx, arg in enumerate(call.args):
                            if arg == inp:
                                checked_type = call.op.checked_type.arg_types[idx]
                        assert checked_type is not None
                        if new_dtype is None:
                            new_dtype = checked_type
                        else:
                            assert new_dtype == checked_type
                    assert new_dtype is not None

                    bypass_quant_dict[inp] = relay.Var(str(inp.name_hint), new_dtype)

        if len(quant_info) != 0:
            new_func = CallerOpInjector(quant_info, (bypass_quant_dict, bypass_dequant_dict)).visit(
                new_ir_mod["main"]
            )
            new_func = relay.Function(new_func.params, new_func.body)
            new_ir_mod["main"] = new_func
            new_ir_mod = relay.transform.InferType()(new_ir_mod)
            new_ir_mod["main"] = OpCanonicalizer().visit(new_ir_mod["main"])
            new_ir_mod = relay.transform.InferType()(new_ir_mod)
            new_ir_mod["main"] = RedundantTupleCleaner().visit(new_ir_mod["main"])
        return relay.transform.InferType()(new_ir_mod)
