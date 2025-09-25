# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Build all the Compass subgraphs."""
import re
import json
import tvm
from tvm import relax, ir, tir
from tvm.relax.expr_functor import PyExprMutator, mutator
from .utils import is_compass_func
from ..config import CompassFunctionConfig, CompassConfig
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
            # The scale and zero point used by Compass have small difference with those of TVM.
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


class _SubgraphBuilder:
    """Build the given Compass subgraph and store the result in function attribute."""

    def __init__(self, forward_engine_type):
        self._forward_engine_type = forward_engine_type
        self._forward_engine = create_forward_engine(forward_engine_type)

    def build(self, func, target):
        """Build the given Compass subgraph and store the result in function attribute."""
        # 1. Process the Compass subgraph through the specific forward engine.
        new_func = self._forward_engine.pre_build(func)
        new_func = new_func.with_attr("compass.pre_build", self._forward_engine_type)
        new_func = new_func.with_attr("target", target)

        # 2. Get quantization information of input and output nodes.
        in_quant_info, out_quant_info = None, None
        # Only "opt_float" don't need to inject nodes and change the input and output type of the
        # Compass subgraph, otherwise obtain them from quantized Compass IR.
        if self._forward_engine_type != "opt_float":
            cfg = CompassFunctionConfig(new_func.attrs.global_symbol)
            in_quant_info, out_quant_info = _parse_in_out_quant_info(cfg.quant_compass_ir_path[0])

        # 3. Change the type of each input node if needed.
        new_params = new_func.params
        new_params_sinfo = [x.struct_info for x in new_func.params]
        if in_quant_info:
            new_params = []
            new_params_sinfo = []
            assert len(in_quant_info) == len(func.params)
            for old_param, quant_info in zip(func.params, in_quant_info):
                param_sinfo = relax.TensorStructInfo(old_param.struct_info.shape, quant_info.dtype)
                new_params_sinfo.append(param_sinfo)
                new_params.append(relax.expr.Var(old_param.name_hint, param_sinfo, old_param.span))

        # 4. Change the type of each output node if needed.
        new_ret_type = func.ret_struct_info
        if out_quant_info:
            is_multiple_output = isinstance(new_ret_type, relax.struct_info.TupleStructInfo)
            old_types = new_ret_type.fields if is_multiple_output else (new_ret_type,)

            new_types = []
            assert len(out_quant_info) == len(old_types)
            for old_type, quant_info in zip(old_types, out_quant_info):
                new_types.append(relax.TensorStructInfo(old_type.shape, quant_info.dtype))

            new_ret_type = relax.TupleStructInfo(new_types) if is_multiple_output else new_types[0]

        # 5. Shrink the original Compass subgraph to one node and as the body of the new function.
        new_body = relax.expr.Var("PlaceHolder", new_ret_type)
        new_func = relax.Function(new_params, new_body, new_ret_type, attrs=new_func.attrs)
        func_sinfo = relax.FuncStructInfo(new_params_sinfo, new_ret_type)
        return new_func, func_sinfo, in_quant_info, out_quant_info


@mutator
class CallerOpInjector(PyExprMutator):
    """Inject necessary quantization or type conversion nodes to the caller."""

    def __init__(self, quant_info, gv2new_gv, mod, bypass_output_dequant):
        super().__init__(mod)
        self._quant_info = quant_info
        self._gv2new_gv = gv2new_gv
        self._bypass_dequant = bypass_output_dequant

    def visit_call_(self, call):
        ret = super().visit_call_(call)
        if ret.op not in self._quant_info:
            return ret

        # This node is a call to a changed Compass subgraph.
        in_quant_info, out_quant_info = self._quant_info[ret.op]

        # Inject necessary nodes after the corresponding arguments.
        new_args = []
        assert len(in_quant_info) == len(ret.args)
        for i, (old_arg, quant_info) in enumerate(zip(ret.args, in_quant_info)):
            old_dtype = call.args[i].checked_type.dtype
            quant_dtype = quant_info.dtype

            # Maybe need insert type conversion node for the int or uint parameter.
            if old_dtype.type_code in (tvm.DataTypeCode.INT, tvm.DataTypeCode.UINT):
                if old_dtype == quant_dtype:
                    new_args.append(old_arg)
                else:
                    a_min = float(tir.min_value(quant_dtype).value)
                    a_max = float(tir.max_value(quant_dtype).value)
                    arg = relax.op.astype(relax.op.clip(old_arg, a_min, a_max), quant_dtype)
                    new_args.append(arg)
                continue

            # Need insert quantize node for the float parameter.
            assert str(old_dtype) in ("float32", "float64")
            if str(old_dtype) == "float64":
                old_arg = relax.op.astype(old_arg, "float32")

            scale = relax.const(quant_info.scale, "float32")
            zp = relax.const(quant_info.zero_point, "int32")  # pylint: disable=invalid-name
            quant = relax.op.qdq.quantize(old_arg, scale, zp, out_dtype=quant_info.dtype)
            new_args.append(self.builder_.emit(quant))

        # Create the new call with the new arguments.
        new_call = self.builder_.emit(relax.Call(self._gv2new_gv[ret.op], new_args))

        # Inject necessary nodes after the corresponding output nodes.
        old_ret_type = call.struct_info
        is_multiple_output = isinstance(old_ret_type, relax.TupleStructInfo)
        old_types = old_ret_type.fields if is_multiple_output else (old_ret_type,)

        out_exprs = []
        assert len(out_quant_info) == len(old_types)
        for i, quant_info in enumerate(out_quant_info):
            out_expr = relax.TupleGetItem(new_call, i) if is_multiple_output else new_call
            old_dtype = old_types[i].dtype

            if i in self._bypass_dequant:
                out_exprs.append(out_expr)
                continue

            # Maybe need insert type conversion node for the int or uint output node.
            if old_dtype.type_code in (tvm.DataTypeCode.INT, tvm.DataTypeCode.UINT):
                if old_dtype == quant_info.dtype:
                    out_exprs.append(out_expr)
                else:
                    a_min = float(tir.min_value(old_dtype).value)
                    a_max = float(tir.max_value(old_dtype).value)
                    out = relax.op.astype(relax.op.clip(out_expr, a_min, a_max), old_dtype)
                    out_exprs.append(out)
                continue

            # Need insert dequantize node for the float output node.
            assert str(old_dtype) in ("float16", "float32", "float64")
            scale = relax.const(quant_info.scale, "float32")
            zero_point = relax.const(quant_info.zero_point, "int32")
            if str(old_dtype) != "float64":
                out_exprs.append(
                    relax.op.qdq.dequantize(out_expr, scale, zero_point, -1, old_dtype)
                )
            else:
                deq = relax.op.qdq.dequantize(out_expr, scale, zero_point, -1, "float32")
                out_exprs.append(relax.op.astype(deq, "float64"))

        return relax.Tuple(out_exprs) if is_multiple_output else out_exprs[0]


@mutator
class OpCanonicalizer(PyExprMutator):
    """Canonicalize some invalid operators."""

    def visit_call_(self, call):
        post = super().visit_call_(call)
        if isinstance(post.op, ir.Op) and post.op.name == "relax.clip":
            clip = post
            if not all(isinstance(x, relax.PrimValue) for x in clip.args[1:]):
                return post
            a_min = float(clip.args[1].value)
            a_max = float(clip.args[2].value)
            dtype = call.struct_info.dtype
            valid_min = float(tir.min_value(dtype).value)
            valid_max = float(tir.max_value(dtype).value)

            # Useless clip, just delete it.
            if a_min <= valid_min and a_max >= valid_max:
                return post.args[0]

            # Normal clip, just keep it.
            if a_min >= valid_min and a_max <= valid_max:
                return post

            # Useful clip but "a_min" or "a_max" need to be adjusted.
            new_min = max(a_min, valid_min)
            new_max = min(a_max, valid_max)

            return relax.op.clip(post.args[0], new_min, new_max)
        return post


@ir.transform.module_pass(opt_level=0)
class BuildCompassSubgraph:
    """Build the Compass subgraphs, store the result in the attribute of the corresponding function,
    shrink the body of each Compass subgraph to one node and inject necessary quantization or type
    conversion nodes for its input and output nodes."""

    def __init__(self, forward_engine, target, bypass_output_dequant):
        self._builder = _SubgraphBuilder(forward_engine)
        self._target = target
        self._bypass_output_dequant = bypass_output_dequant

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        # 1. Build all Compass subgraphs, and change each one according to quantization.
        is_quant_model = CompassConfig.get().common["compat_quantized_model"] == "true"
        quant_info = {}
        gv2new_gv = {}
        for gvar, func in ir_mod.functions.items():
            if not is_compass_func(func):
                continue
            new_func, func_sinfo, in_qinfo, out_qinfo = self._builder.build(func, self._target)
            if not is_quant_model and in_qinfo and out_qinfo:
                new_gvar = relax.GlobalVar(gvar.name_hint)
                relax.expr._update_struct_info(new_gvar, func_sinfo)
                del ir_mod[gvar]
                ir_mod[new_gvar] = new_func
                gv2new_gv[gvar] = new_gvar
                quant_info[gvar] = (in_qinfo, out_qinfo)
            else:
                ir_mod[gvar] = new_func

        # 2. Inject necessary quantization or type conversion nodes in caller.
        if len(quant_info) != 0:
            for gvar, func in ir_mod.functions.items():
                if is_compass_func(func):
                    continue
                new_func = CallerOpInjector(
                    quant_info, gv2new_gv, ir_mod, self._bypass_output_dequant
                ).visit_expr(func)
                new_func = OpCanonicalizer().visit_expr(new_func)
                ir_mod[gvar] = new_func
                ir_mod = relax.transform.RemoveUnusedOutputs()(ir_mod)
        return ir_mod
