# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""AIPU Compass IR codegen of Relay."""
import os
import textwrap
from collections.abc import Iterable
import numpy as np
from tvm import ir, relay, tir, nd
from tvm.relay.op.contrib.aipu_compass import CODEGEN_CUSTOM_OP_DICT
from tvm.relay.op.contrib.aipu_compass.utils import get_activation_str, unpack_commutative_args
from tvm.relay.op.contrib.aipu_compass.utils import peel_hardswish
from ..runtime import AipuCompassBasicConfig


def _is_composite_op(expr, name):
    if (
        isinstance(expr, relay.Function)
        and "Composite" in expr.attrs
        and expr.attrs["Composite"] == name
    ):
        return True
    return False


def _convert_composite_arg(arg, call):
    if isinstance(call, relay.Call) and isinstance(call.op, relay.Function):
        func_params = list(call.op.params)
        assert arg in func_params
        return call.args[func_params.index(arg)]
    return arg


def _is_custom_single_op(op):
    if isinstance(op, ir.Op) and op.name in CODEGEN_CUSTOM_OP_DICT.keys():
        return True
    return False


def _is_composite_custom_op(expr):
    if (
        isinstance(expr, relay.Function)
        and "Composite" in expr.attrs
        and expr.attrs["Composite"] in CODEGEN_CUSTOM_OP_DICT.keys()
    ):
        return True
    return False


def _verify_shape(shape):
    if isinstance(shape, (tir.IntImm, int)):
        return int(shape)
    if isinstance(shape, (Iterable, ir.container.Array)):
        return [_verify_shape(val) for val in shape]
    raise RuntimeError(f"Unsupported shape type: {type(shape)}")


class SetQnnParams(relay.ExprVisitor):
    """Set qnn params of all nodes."""

    def __init__(self):
        super().__init__()
        self._expr2qnn_params = dict()

    def _set_qnn_params(self, expr, scale, zero_point, call=None):
        qnn_params = self._get_qnn_params(expr)
        if qnn_params:
            assert (
                qnn_params[0].data.numpy() == scale.data.numpy()
                and qnn_params[1].data.numpy() == zero_point.data.numpy()
            ), "Qnn params are not equal when set this expr again."
        else:
            expr = _convert_composite_arg(expr, call)
            self._expr2qnn_params[expr] = (scale, zero_point)

    def _get_qnn_params(self, expr, call=None):
        return self._expr2qnn_params.get(_convert_composite_arg(expr, call), None)

    def get_qnn_params_dict(self, expr):
        self.visit(expr)
        return self._expr2qnn_params

    def visit_tuple(self, tup):
        qnn_params = self._get_qnn_params(tup)
        if qnn_params:
            for i in range(len(tup.fields)):
                if isinstance(qnn_params[0], relay.Tuple):
                    assert (
                        len(tup.fields) == len(qnn_params[0]) == len(qnn_params[1])
                    ), "Different length of input with their scale and zp"
                    self._set_qnn_params(tup.fields[i], qnn_params[0][i], qnn_params[1][i])
                else:
                    self._set_qnn_params(tup.fields[i], *qnn_params)

        for x in tup.fields:
            self.visit(x)

        if not qnn_params:
            arg_qnn_params = self._get_qnn_params(tup.fields[0])
            if arg_qnn_params:
                self._set_qnn_params(tup, *arg_qnn_params)

    def visit_tuple_getitem(self, t):
        qnn_params = self._get_qnn_params(t)
        if qnn_params and not _is_composite_op(t.tuple_value.op, "aipu_compass.QnnBasicLSTM"):
            self._set_qnn_params(t.tuple_value, *qnn_params)

        self.visit(t.tuple_value)

        if not qnn_params:
            arg_qnn_params = self._get_qnn_params(t.tuple_value)
            if arg_qnn_params:
                self._set_qnn_params(t, *arg_qnn_params)

    def visit_call(self, call):
        if _is_composite_op(call.op, "aipu_compass.QnnSoftmax") or _is_composite_op(
            call.op, "aipu_compass.QnnResize2D"
        ):
            func = call.op
            quantize = func.body
            dequantize = quantize.args[0].args[0]
            self._set_qnn_params(dequantize.args[0], dequantize.args[1], dequantize.args[2], call)
            self._set_qnn_params(call, quantize.args[1], quantize.args[2])
        elif (
            call.op == relay.op.get("qnn.add")
            or call.op == relay.op.get("qnn.mul")
            or call.op == relay.op.get("qnn.subtract")
        ):
            self._set_qnn_params(call.args[0], call.args[2], call.args[3], call)
            self._set_qnn_params(call.args[1], call.args[4], call.args[5], call)
            self._set_qnn_params(call, call.args[6], call.args[7])
        elif (
            call.op == relay.op.get("qnn.tanh")
            or call.op == relay.op.get("qnn.sigmoid")
            or call.op == relay.op.get("qnn.rsqrt")
            or call.op == relay.op.get("qnn.requantize")
        ):
            self._set_qnn_params(call.args[0], call.args[1], call.args[2], call)
            self._set_qnn_params(call, call.args[3], call.args[4])
        elif _is_composite_op(call.op, "aipu_compass.QnnConvolution2D"):
            func = call.op
            scale, zero_point = None, None
            if func.body.op == relay.op.get("qnn.requantize"):
                requantize = func.body
                if requantize.args[0].op == relay.op.get("qnn.leaky_relu"):
                    scale, zero_point = requantize.args[3:]
                    requantize = requantize.args[0].args[0]
            elif func.body.op == relay.op.get("qnn.leaky_relu"):
                leaky_relu = func.body
                requantize = leaky_relu.args[0]
                scale, zero_point = leaky_relu.args[3:]
            elif func.body.op == relay.op.get("clip"):
                requantize = func.body.args[0]

            if requantize.args[0].op == relay.op.get("add"):
                add = requantize.args[0]
                qnn_conv2d, _ = unpack_commutative_args(add)
            else:
                add = None
                qnn_conv2d = requantize.args[0]
            if not scale and not zero_point:
                scale, zero_point = requantize.args[3:]

            input_tensor = qnn_conv2d.args[0]
            self._set_qnn_params(input_tensor, qnn_conv2d.args[4], qnn_conv2d.args[2], call)
            self._set_qnn_params(call, scale, zero_point)
        elif _is_composite_op(call.op, "aipu_compass.QnnMatmul"):
            expand_dims = call.op.body
            requantize = expand_dims.args[0]
            qdense = requantize.args[0]
            transpose = qdense.args[1]
            reshape1 = transpose.args[0]
            input0 = qdense.args[0].args[0]
            input1 = reshape1.args[0]
            self._set_qnn_params(input0, qdense.args[4], qdense.args[2], call)
            self._set_qnn_params(input1, qdense.args[5], qdense.args[3], call)
            self._set_qnn_params(call, requantize.args[3], requantize.args[4])
        elif _is_composite_op(call.op, "aipu_compass.QnnEltwiseRelu"):
            func = call.op
            clip = func.body
            qnn_eltwise = clip.args[0]
            self._set_qnn_params(
                qnn_eltwise.args[0], qnn_eltwise.args[2], qnn_eltwise.args[3], call
            )
            self._set_qnn_params(
                qnn_eltwise.args[1], qnn_eltwise.args[4], qnn_eltwise.args[5], call
            )
            self._set_qnn_params(call, qnn_eltwise.args[6], qnn_eltwise.args[7])
        elif _is_composite_op(call.op, "aipu_compass.QnnDenseAdd"):
            func = call.op
            requantize = func.body
            qnn_dense, _ = unpack_commutative_args(requantize.args[0])
            input_tensor = qnn_dense.args[0]
            self._set_qnn_params(input_tensor, qnn_dense.args[4], qnn_dense.args[2], call)
            self._set_qnn_params(call, requantize.args[3], requantize.args[4])
        elif _is_composite_op(call.op, "aipu_compass.QnnDense"):
            func = call.op
            if func.body.op == relay.op.get("qnn.requantize"):
                requantize = func.body
            elif func.body.op == relay.op.get("clip"):
                clip = func.body
                requantize = clip.args[0]
            qnn_dense = requantize.args[0]
            input_tensor = qnn_dense.args[0]
            self._set_qnn_params(input_tensor, qnn_dense.args[4], qnn_dense.args[2], call)
            self._set_qnn_params(call, requantize.args[3], requantize.args[4])
        elif _is_composite_op(call.op, "aipu_compass.QnnReduce"):
            func = call.op
            requantize = func.body
            reduce_call = requantize.args[0]
            data = reduce_call.args[0].args[0]
            self._set_qnn_params(data, requantize.args[1], requantize.args[2], call)
            self._set_qnn_params(call, requantize.args[3], requantize.args[4])
        elif call.op == relay.op.get("qnn.concatenate"):
            self._set_qnn_params(call.args[0], call.args[1], call.args[2])
            self._set_qnn_params(call, call.args[3], call.args[4])
        elif _is_composite_op(call.op, "aipu_compass.QnnSilu"):
            func = call.op
            composite = func.body
            inp = composite.args[0]
            self._set_qnn_params(inp, composite.args[2], composite.args[3], call)
            self._set_qnn_params(call, composite.args[6], composite.args[7])
        elif _is_composite_op(call.op, "aipu_compass.QnnSquaredDifference"):
            func = call.op
            squared_difference = func.body
            difference = squared_difference.args[0]
            self._set_qnn_params(difference.args[0], difference.args[2], difference.args[3], call)
            self._set_qnn_params(difference.args[1], difference.args[4], difference.args[5], call)
            self._set_qnn_params(call, squared_difference.args[6], squared_difference.args[7])
        elif _is_composite_op(call.op, "aipu_compass.QnnHardSwish"):
            func = call.op
            quantize = func.body
            dequantize, _, _ = peel_hardswish(quantize.args[0])
            self._set_qnn_params(dequantize.args[0], dequantize.args[1], dequantize.args[2], call)
            self._set_qnn_params(call, quantize.args[1], quantize.args[2])
        elif _is_composite_op(call.op, "aipu_compass.QnnRequantS2D"):
            func = call.op
            space_to_depth = func.body
            requant = space_to_depth.args[0]
            self._set_qnn_params(requant.args[0], requant.args[1], requant.args[2], call)
            self._set_qnn_params(call, requant.args[3], requant.args[4])
        elif _is_composite_op(call.op, "aipu_compass.QnnFlattenPrelu"):
            func = call.op
            reshape1 = func.body
            qnn_prelu = reshape1.args[0]
            reshape0 = qnn_prelu.args[0]
            self._set_qnn_params(reshape0.args[0], qnn_prelu.args[2], qnn_prelu.args[3], call)
            self._set_qnn_params(call, qnn_prelu.args[6], qnn_prelu.args[7])
        elif _is_composite_op(call.op, "aipu_compass.QnnCast"):
            func = call.op
            quantize = func.body
            cast = quantize.args[0]
            self._set_qnn_params(cast.args[0], quantize.args[1], quantize.args[2], call)
            self._set_qnn_params(call, quantize.args[1], quantize.args[2])
        elif _is_composite_op(call.op, "aipu_compass.QnnSigmoid"):
            func = call.op
            if func.body.op == relay.op.get("qnn.requantize"):
                requantize = func.body
                sigmoid = requantize.args[0]
                self._set_qnn_params(sigmoid.args[0], sigmoid.args[1], sigmoid.args[2], call)
                self._set_qnn_params(call, requantize.args[3], requantize.args[4])
            elif func.body.op == relay.op.get("qnn.quantize"):
                quantize = func.body
                dequantize = quantize.args[0].args[0]
                self._set_qnn_params(
                    dequantize.args[0], dequantize.args[1], dequantize.args[2], call
                )
                self._set_qnn_params(call, quantize.args[1], quantize.args[2])
        elif _is_composite_op(call.op, "aipu_compass.QnnMinimum"):
            func = call.op
            requantize = func.body
            clip = requantize.args[0]
            minimum = clip.args[0]
            self._set_qnn_params(minimum.args[0], requantize.args[1], requantize.args[2], call)
            self._set_qnn_params(minimum.args[1], requantize.args[1], requantize.args[2], call)
            self._set_qnn_params(call, requantize.args[3], requantize.args[4])
        elif _is_composite_op(call.op, "aipu_compass.QnnMirrorPad"):
            func = call.op
            requantize = func.body
            mirror_pad = requantize.args[0]
            self._set_qnn_params(mirror_pad.args[0], requantize.args[1], requantize.args[2], call)
            self._set_qnn_params(call, requantize.args[3], requantize.args[4])
        elif _is_composite_op(call.op, "aipu_compass.QnnBasicLSTM"):
            func = call.op
            func_out = func.body
            assert isinstance(func_out, relay.Tuple), "the output of basiclstm must be a tuple."
            mul_hout, add_cout = func_out
            mul_sp2 = add_cout.args[0]
            sigmoid_sp1, initial_c = mul_sp2.args[:2]
            split = sigmoid_sp1.args[0].tuple_value
            add_bias = split.args[0]
            add_fc = add_bias.args[0]
            req_after_squeeze, req_after_hstate = add_fc.args[:2]
            q_dense_after_squeeze = req_after_squeeze.args[0]
            q_dense_after_hstate = req_after_hstate.args[0]
            input_x = q_dense_after_squeeze.args[0]
            initial_h = q_dense_after_hstate.args[0]
            self._set_qnn_params(
                input_x, q_dense_after_squeeze.args[4], q_dense_after_squeeze.args[2], call
            )
            self._set_qnn_params(
                initial_h, q_dense_after_hstate.args[4], q_dense_after_hstate.args[2], call
            )
            self._set_qnn_params(initial_c, mul_sp2.args[4], mul_sp2.args[5], call)
            mul_hout_scale, mul_hout_zp = mul_hout.args[6:8]
            qnn_add_scale, qnn_add_zp = add_cout.args[6:8]
            self._set_qnn_params(mul_hout, mul_hout_scale, mul_hout_zp)
            self._set_qnn_params(add_cout, qnn_add_scale, qnn_add_zp)
        else:
            qnn_params = self._get_qnn_params(call)
            if qnn_params:
                for arg in call.args:
                    self._set_qnn_params(arg, *qnn_params)

        for arg in call.args:
            self.visit(arg)

        qnn_params = self._get_qnn_params(call)
        if not qnn_params:
            arg_qnn_params = self._get_qnn_params(call.args[0])
            if arg_qnn_params:
                self._set_qnn_params(call, *arg_qnn_params)


# Inherit from "relay.ExprFunctor" instead of "relay.ExprVisitor" for avoiding
# missing error report of the unimplemented AST nodes.
# pylint: disable=abstract-method
class CodeGenAipuCompass(relay.ExprFunctor):
    """AIPU Compass generator of Relay IR."""

    def __init__(self):
        super().__init__()
        self._ir_text = ""
        self._np_arraies = []
        self._layer_count = 0
        self._expr2var_name = dict()
        self._temp_var_idx = 0
        self._unused_func_params = None
        self._const2offset_nbytes = dict()
        self._cur_offset = 0
        self._constant_codegen = []
        self._tuple2tgn = dict()
        self._expr2qnn_params = dict()

    def gen(self, func):
        if AipuCompassBasicConfig.get().common.get("compat_quantized_model") == "true":
            self._expr2qnn_params = SetQnnParams().get_qnn_params_dict(func)
        self.visit(func)
        ir_bin = bytearray()
        for np_arr in self._np_arraies:
            ir_bin += np_arr.tobytes()
        return self._ir_text, np.frombuffer(ir_bin, dtype="uint8")

    def gen2file(self, func, txt_path, bin_path):
        ir_txt, ir_bin = self.gen(func)
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        open(txt_path, "w", encoding="utf-8").write(ir_txt)
        os.makedirs(os.path.dirname(bin_path), exist_ok=True)
        ir_bin.tofile(bin_path)

    def _get_or_alloc_var_name(self, expr):
        if expr not in self._expr2var_name:
            if isinstance(expr, relay.Var):
                name = expr.name_hint
                assert name not in self._expr2var_name.values()
            else:
                name = f"temp_var_{self._temp_var_idx}"
                self._temp_var_idx += 1

            self._expr2var_name[expr] = name

        return self._expr2var_name[expr]

    def _get_qnn_params(self, expr, call=None, is_to_list=False):
        qnn_params = self._expr2qnn_params.get(_convert_composite_arg(expr, call), None)
        if qnn_params and is_to_list:
            scale = qnn_params[0].data.numpy().reshape(-1).tolist()
            zero_point = qnn_params[1].data.numpy().reshape(-1).tolist()
            qnn_params = (scale, zero_point)
        return qnn_params

    def _get_layer_idx(self):
        layer_idx = self._layer_count
        self._layer_count += 1
        return layer_idx

    def _gen_basic_layer_items(self, layer_type, inputs, call):
        if not isinstance(inputs, (list, tuple, ir.container.Array)):
            inputs = [inputs]
        input_names = []
        input_shapes = []
        input_types = []
        for arg in inputs:
            arg = _convert_composite_arg(arg, call)
            if isinstance(arg, relay.Constant):
                self._gen_constant(arg)
            if isinstance(arg, relay.Tuple):
                for item in arg.fields:
                    if isinstance(item, relay.Constant):
                        self._gen_constant(item)
                    input_names.append(self._get_or_alloc_var_name(item))
                for item in arg.checked_type.fields:
                    input_shapes.append(item.shape)
                    input_types.append(item.dtype)
            else:
                arg_type = arg.checked_type
                if isinstance(arg_type, ir.TupleType):
                    for i, field in enumerate(arg_type.fields):
                        input_names.append(arg.name_hint + "_" + str(i))
                        input_shapes.append(field.shape)
                        input_types.append(field.dtype)
                else:
                    input_names.append(self._get_or_alloc_var_name(arg))
                    input_shapes.append(arg.checked_type.shape)
                    input_types.append(arg.checked_type.dtype)

        output_names = []
        output_shapes = []
        output_types = []
        output_scales = []
        output_zps = []
        if isinstance(call.checked_type, relay.TupleType):
            tgn_dict = {i.index: i for i in self._tuple2tgn[call]}
            for i, field in enumerate(call.checked_type.fields):
                if i in tgn_dict.keys():
                    out = tgn_dict[i]
                    output_names.append(self._get_or_alloc_var_name(out))
                    qnn_params = self._get_qnn_params(out, is_to_list=True)
                    if qnn_params:
                        scale, zero_point = qnn_params
                        output_scales.append(scale)
                        output_zps.append(zero_point)
                else:
                    output_names.append("useless_out_" + str(self._get_layer_idx()) + str(i))
                output_shapes.append(field.shape)
                output_types.append(field.dtype)
        else:
            output_names.append(self._get_or_alloc_var_name(call))
            output_shapes.append(call.checked_type.shape)
            output_types.append(call.checked_type.dtype)
            qnn_params = self._get_qnn_params(call, is_to_list=True)
            if qnn_params:
                scale, zero_point = qnn_params
                output_scales.append(scale)
                output_zps.append(zero_point)

        layer_idx = self._get_layer_idx()
        input_types = [inp_type if inp_type != "bool" else "uint8" for inp_type in input_types]
        output_types = [out_type if out_type != "bool" else "uint8" for out_type in output_types]

        self._ir_text += textwrap.dedent(
            f"""
            layer_id={layer_idx}
            layer_name={layer_idx}_{layer_type.lower()}
            layer_type={layer_type}
            layer_bottom=[{", ".join(input_names)}]
            layer_bottom_shape={_verify_shape(input_shapes)}
            layer_bottom_type=[{", ".join(input_types)}]
            layer_top=[{", ".join(output_names)}]
            layer_top_shape={_verify_shape(output_shapes)}
            layer_top_type=[{", ".join(output_types)}]"""
        )
        if output_scales and output_zps:
            self._ir_text += textwrap.dedent(
                f"""
                layer_top_scale={output_scales}
                layer_top_zp={output_zps}"""
            )

    def _gen_constant(self, const_node):
        if const_node in self._constant_codegen:
            return

        weight_ttype = const_node.checked_type
        weight_offset, weight_nbytes = self._get_offset_nbytes(const_node)
        layer_idx = self._get_layer_idx()
        output_names = self._get_or_alloc_var_name(const_node)
        weight_dtype = "uint8" if weight_ttype.dtype == "bool" else weight_ttype.dtype
        weight_shape = _verify_shape(weight_ttype.shape)

        self._ir_text += textwrap.dedent(
            f"""
            layer_id={layer_idx}
            layer_name={layer_idx}_constant
            layer_type=Constant
            layer_bottom=[]
            layer_bottom_shape=[]
            layer_bottom_type=[]
            layer_top=[{output_names}]
            layer_top_shape=[{weight_shape}]
            layer_top_type=[{weight_dtype}]"""
        )

        qnn_params = self._get_qnn_params(const_node, is_to_list=True)
        if qnn_params:
            scale, zero_point = qnn_params
            self._ir_text += textwrap.dedent(
                f"""
                layer_top_scale=[{scale}]
                layer_top_zp=[{zero_point}]"""
            )

        self._ir_text += textwrap.dedent(
            f"""
            weights_type={weight_dtype}
            weights_offset={weight_offset}
            weights_size={weight_nbytes}
            weights_shape={_verify_shape(weight_ttype.shape)}"""
        )

        if qnn_params:
            scale_const, zp_const = self._get_qnn_params(const_node)
            scale_offset, scale_nbytes = self._get_offset_nbytes(scale_const)
            zp_offset, zp_nbytes = self._get_offset_nbytes(zp_const)

            self._ir_text += textwrap.dedent(
                f"""
                weights_scale_type=float32
                weights_scale_offset={scale_offset}
                weights_scale_size={scale_nbytes}
                weights_scale_shape={_verify_shape(scale_const.checked_type.shape)}
                weights_zp_type=int32
                weights_zp_offset={zp_offset}
                weights_zp_size={zp_nbytes}
                weights_zp_shape={_verify_shape(zp_const.checked_type.shape)}"""
            )

        self._ir_text += textwrap.dedent("\n")
        self._constant_codegen.append(const_node)

    def visit_var(self, var):
        # The free variable or local variable shouldn't exist.
        assert var in self._unused_func_params
        self._unused_func_params.remove(var)

    def visit_tuple(self, tup):
        for x in tup.fields:
            self.visit(x)

    def visit_tuple_getitem(self, t):
        self.visit(t.tuple_value)

    def visit_constant(self, const_node):
        pass

    def _get_offset_nbytes(self, const_node):
        offset_nbytes = self._const2offset_nbytes.get(const_node, None)
        if not offset_nbytes:
            np_arr = const_node.data.numpy()
            self._np_arraies.append(np_arr)
            offset_nbytes = (self._cur_offset, np_arr.nbytes)
            self._const2offset_nbytes[const_node] = offset_nbytes
            self._cur_offset += np_arr.nbytes
        return offset_nbytes

    def visit_function(self, func):
        def _analysis_multiply_output(t):
            if isinstance(t, relay.TupleGetItem):
                tuple_value = t.tuple_value
                if tuple_value not in self._tuple2tgn:
                    self._tuple2tgn[tuple_value] = [t]
                else:
                    self._tuple2tgn[tuple_value].append(t)

        relay.analysis.post_order_visit(func.body, _analysis_multiply_output)
        for tuple_expr in self._tuple2tgn:
            self._tuple2tgn[tuple_expr].sort(key=lambda getitem: int(getitem.index))

        input_names = []
        input_ir_texts = []

        def _gen_input(param, ttype=None):
            idx = self._get_layer_idx()
            if not ttype:
                ttype = param.checked_type
            dtype = "uint8" if ttype.dtype == "bool" else ttype.dtype
            name = self._get_or_alloc_var_name(param)
            input_names.append(name)

            input_ir_texts.append(
                (
                    param,
                    textwrap.dedent(
                        f"""
                layer_id={idx}
                layer_name={idx}_input
                layer_type=Input
                layer_bottom=[]
                layer_bottom_shape=[]
                layer_bottom_type=[]
                layer_top=[{name}]
                layer_top_shape=[{_verify_shape(ttype.shape)}]
                layer_top_type=[{dtype}]"""
                    ),
                )
            )

        for param in func.params:
            if isinstance(param.checked_type, ir.TupleType):
                for i, field in enumerate(param.checked_type.fields):
                    name = param.name_hint + "_" + str(i)
                    _gen_input(relay.Var(name, field), field)
            else:
                _gen_input(param)

        self._unused_func_params = list(func.params)
        self.visit(func.body)
        assert len(self._unused_func_params) == 0, f"Unused parameters: {self._unused_func_params}."

        if isinstance(func.body, relay.Tuple):
            outputs = list(func.body.fields)
        elif isinstance(func.body, relay.TupleWrapper):
            outputs = list(func.body.tuple_value.fields)
        else:
            outputs = [func.body]
        out_names = [self._expr2var_name[expr] for expr in outputs]

        compat_quantized_model = AipuCompassBasicConfig.get().common.get(
            "compat_quantized_model", "false"
        )

        header = textwrap.dedent(
            f"""
            model_name=unknown
            layer_number={self._layer_count}
            data_format=NHWC
            precision=float32
            batch_size=1
            input_tensors=[{",".join(input_names)}]
            output_tensors=[{",".join(out_names)}]
            compat_quantized_model={compat_quantized_model}
            """
        )

        body_ir_text = self._ir_text
        self._ir_text = header
        for param, input_ir_text in input_ir_texts:
            qnn_params = self._get_qnn_params(param, is_to_list=True)
            if qnn_params:
                scale, zero_point = qnn_params
                input_ir_text += textwrap.dedent(
                    f"""
                    layer_top_scale=[{scale}]
                    layer_top_zp=[{zero_point}]"""
                )
            input_ir_text += textwrap.dedent("\n")
            self._ir_text += input_ir_text

        self._ir_text += body_ir_text

    def visit_call(self, call):
        for arg in call.args:
            self.visit(arg)

        if _is_composite_op(call.op, "aipu_compass.Convolution2D"):
            self._gen_convolution2d(call)
        elif call.op == relay.op.get("nn.conv2d"):
            self._gen_convolution2d(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnConvolution2D"):
            self._gen_qnn_convolution2d(call)
        elif _is_composite_op(call.op, "aipu_compass.Convolution3D"):
            self._gen_convolution3d(call)
        elif _is_composite_op(call.op, "aipu_compass.ElementwiseRelu"):
            self._gen_elementwise_relu(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnMinimum"):
            self._gen_qnn_minimum(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnEltwiseRelu"):
            self._gen_qnn_eltwise_relu(call)
        elif _is_composite_op(call.op, "aipu_compass.DenseAdd"):
            self._gen_dense_add(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnDenseAdd"):
            self._gen_qnn_dense_add(call)
        elif _is_composite_op(call.op, "aipu_compass.Dense"):
            self._gen_dense(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnDense"):
            self._gen_qnn_dense(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnMatmul"):
            self._gen_qnn_matmul(call)
        elif _is_composite_op(call.op, "aipu_compass.BatchNorm"):
            self._gen_batchnorm(call)
        elif _is_composite_op(call.op, "aipu_compass.HardSwish"):
            self._gen_hardswish(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnHardSwish"):
            self._gen_qnn_hardswish(call)
        elif _is_composite_op(call.op, "aipu_compass.Softplus"):
            self._gen_softplus(call)
        elif _is_composite_op(call.op, "aipu_compass.Gelu"):
            self._gen_gelu(call)
        elif _is_composite_op(call.op, "aipu_compass.MeanVarianceNormalization"):
            self._gen_mean_variance_norm(call)
        elif _is_composite_op(call.op, "aipu_compass.LayerNorm0"):
            self._gen_layernorm0(call)
        elif _is_composite_op(call.op, "aipu_compass.LayerNorm1"):
            self._gen_layernorm1(call)
        elif _is_composite_op(call.op, "aipu_compass.LogSoftmax"):
            self._gen_log_softmax(call)
        elif _is_composite_op(call.op, "aipu_compass.InstanceNorm"):
            self._gen_instancenorm(call)
        elif _is_composite_op(call.op, "aipu_compass.L2Norm"):
            self._gen_l2norm(call)
        elif (
            call.op == relay.op.get("add")
            or call.op == relay.op.get("subtract")
            or call.op == relay.op.get("multiply")
            or call.op == relay.op.get("maximum")
            or call.op == relay.op.get("minimum")
        ):
            self._gen_elementwise(call)
        elif (
            call.op == relay.op.get("qnn.add")
            or call.op == relay.op.get("qnn.mul")
            or call.op == relay.op.get("qnn.subtract")
        ):
            self._gen_qnn_elementwise(call)
        elif call.op == relay.op.get("divide"):
            self._gen_divide(call)
        elif call.op == relay.op.get("copy"):
            self._gen_copy(call)
        elif call.op == relay.op.get("mod"):
            self._gen_mod(call)
        elif call.op == relay.op.get("nn.max_pool2d"):
            self._gen_max_pooling2d(call)
        elif call.op == relay.op.get("nn.avg_pool2d"):
            self._gen_avg_pooling2d(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnAvgPool2D"):
            self._gen_qnn_avg_pool2d(call)
        elif call.op == relay.op.get("nn.global_avg_pool2d"):
            self._gen_global_pooling2d(call, "AVG")
        elif call.op == relay.op.get("nn.global_max_pool2d"):
            self._gen_global_pooling2d(call, "MAX")
        elif (
            call.op == relay.op.get("all")
            or call.op == relay.op.get("any")
            or call.op == relay.op.get("max")
            or call.op == relay.op.get("min")
            or call.op == relay.op.get("prod")
            or call.op == relay.op.get("sum")
            or call.op == relay.op.get("mean")
            or call.op == relay.op.get("variance")
        ):
            self._gen_reduce(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnReduce"):
            self._gen_qnn_reduce(call)
        elif call.op == relay.op.get("concatenate"):
            self._gen_concatenate(call)
        elif call.op == relay.op.get("qnn.concatenate"):
            self._gen_qnn_concatenate(call)
        elif call.op == relay.op.get("reshape"):
            self._gen_reshape(call)
        elif call.op == relay.op.get("nn.softmax"):
            self._gen_softmax(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnSoftmax"):
            self._gen_qnn_softmax(call)
        elif call.op == relay.op.get("nn.log_softmax"):
            self._gen_log_softmax(call)
        elif call.op == relay.op.get("transpose"):
            self._gen_transpose(call)
        elif call.op == relay.op.get("one_hot"):
            self._gen_one_hot(call)
        elif call.op == relay.op.get("image.resize2d"):
            self._gen_resize(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnResize2D"):
            self._gen_qnn_resize(call)
        elif call.op == relay.op.get("image.grid_sample"):
            self._gen_grid_sample(call)
        elif call.op == relay.op.get("split"):
            self._gen_split(call)
        elif call.op == relay.op.get("sigmoid"):
            self._gen_sigmoid(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnSigmoid"):
            self._gen_qnn_sigmoid_req_q(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnBasicLSTM"):
            self._gen_qnn_basiclstm(call)
        elif call.op == relay.op.get("qnn.sigmoid"):
            self._gen_qnn_sigmoid(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnSilu"):
            self._gen_qnn_silu(call)
        elif call.op == relay.op.get("nn.pad"):
            self._gen_pad(call)
        elif call.op == relay.op.get("exp"):
            self._gen_exp(call)
        elif call.op == relay.op.get("erf"):
            self._gen_erf(call)
        elif call.op == relay.op.get("power"):
            self._gen_simple_op(call, "Pow")
        elif call.op == relay.op.get("strided_slice"):
            self._gen_strided_slice(call)
        elif call.op == relay.op.get("tile"):
            self._gen_tile(call)
        elif call.op == relay.op.get("nn.relu"):
            self._gen_relu(call)
        elif call.op == relay.op.get("nn.leaky_relu"):
            self._gen_leaky_relu(call)
        elif call.op == relay.op.get("clip"):
            self._gen_clip(call)
        elif call.op == relay.op.get("cast"):
            self._gen_cast(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnCast") or call.op == relay.op.get(
            "qnn.requantize"
        ):
            self._gen_qnn_cast(call)
        elif call.op == relay.op.get("nn.lrn"):
            self._gen_lrn(call)
        elif call.op == relay.op.get("argmax"):
            self._gen_argminmax(call, "MAX")
        elif call.op == relay.op.get("argmin"):
            self._gen_argminmax(call, "MIN")
        elif call.op == relay.op.get("log"):
            self._gen_log(call)
        elif call.op == relay.op.get("abs"):
            self._gen_abs(call)
        elif call.op == relay.op.get("cos"):
            self._gen_cos(call)
        elif call.op == relay.op.get("tanh"):
            self._gen_tanh(call)
        elif call.op == relay.op.get("qnn.tanh"):
            self._gen_qnn_tanh(call)
        elif call.op == relay.op.get("reverse_sequence"):
            self._gen_reverse_sequence(call)
        elif call.op == relay.op.get("nn.batch_matmul"):
            self._gen_batch_matmul(call)
        elif call.op == relay.op.get("negative"):
            self._gen_negative(call)
        elif call.op == relay.op.get("nn.space_to_depth"):
            self._gen_space_to_depth(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnRequantS2D"):
            self._gen_qnn_requant_s2d(call)
        elif call.op == relay.op.get("nn.depth_to_space"):
            self._gen_depth_to_space(call)
        elif call.op == relay.op.get("nn.batch_to_space_nd"):
            self._gen_batch_to_space_nd(call)
        elif call.op == relay.op.get("nn.space_to_batch_nd"):
            self._gen_space_to_batch_nd(call)
        elif call.op == relay.op.get("contrib.aipu_compass.gruv3"):
            self._gen_gruv3(call)
        elif call.op == relay.op.get("nn.prelu"):
            self._gen_prelu(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnFlattenPrelu"):
            self._gen_qnn_flatten_prelu(call)
        elif call.op == relay.op.get("take"):
            self._gen_take(call)
        elif call.op == relay.op.get("equal"):
            self._gen_logical(call, "EQUAL")
        elif call.op == relay.op.get("not_equal"):
            self._gen_logical(call, "NOT_EQUAL")
        elif call.op == relay.op.get("greater"):
            self._gen_logical(call, "GREATER")
        elif call.op == relay.op.get("greater_equal"):
            self._gen_logical(call, "GREATER_EQUAL")
        elif call.op == relay.op.get("less"):
            self._gen_logical(call, "LESS")
        elif call.op == relay.op.get("less_equal"):
            self._gen_logical(call, "LESS_EQUAL")
        elif call.op == relay.op.get("logical_and"):
            self._gen_logical(call, "AND")
        elif call.op == relay.op.get("logical_not"):
            self._gen_logical(call, "NOT")
        elif call.op == relay.op.get("logical_or"):
            self._gen_logical(call, "OR")
        elif call.op == relay.op.get("logical_xor"):
            self._gen_logical(call, "XOR")
        elif call.op == relay.op.get("topk"):
            self._gen_topk(call)
        elif call.op == relay.op.get("round"):
            self._gen_round(call)
        elif call.op == relay.op.get("sqrt"):
            self._gen_simple_op(call, "Sqrt")
        elif call.op == relay.op.get("rsqrt"):
            self._gen_simple_op(call, "Rsqrt")
        elif call.op == relay.op.get("qnn.rsqrt"):
            self._gen_simple_qnn_op(call, "Rsqrt")
        elif _is_composite_op(call.op, "aipu_compass.QnnSquaredDifference"):
            self._gen_qnn_squared_difference(call)
        elif call.op == relay.op.get("sign"):
            self._gen_simple_op(call, "Sign")
        elif call.op == relay.op.get("sin"):
            self._gen_simple_op(call, "Sine")
        elif call.op == relay.op.get("tan"):
            self._gen_simple_op(call, "Tan")
        elif call.op == relay.op.get("image.crop_and_resize"):
            self._gen_crop_and_resize(call)
        elif call.op == relay.op.get("scatter_elements"):
            self._gen_scatter_elements(call)
        elif call.op == relay.op.get("scatter_nd"):
            self._gen_scatter_nd(call)
        elif call.op == relay.op.get("gather"):
            self._gen_gather(call)
        elif call.op == relay.op.get("gather_nd"):
            self._gen_gather_nd(call)
        elif call.op == relay.op.get("where"):
            self._gen_where(call)
        elif call.op == relay.op.get("contrib.aipu_compass.ctc_greedy_decoder"):
            self._gen_ctc_greedy_decoder(call)
        elif call.op == relay.op.get("contrib.aipu_compass.fake_quant_with_min_max_vars"):
            self._gen_fake_quant_min_max_vars(call)
        elif call.op == relay.op.get("contrib.aipu_compass.channel_shuffle"):
            self._gen_channel_shuffle(call)
        elif call.op == relay.op.get("contrib.aipu_compass.divide_mod"):
            self._gen_divide_mod(call)
        elif call.op == relay.op.get("nn.mirror_pad"):
            self._gen_mirror_pad(call)
        elif _is_composite_op(call.op, "aipu_compass.QnnMirrorPad"):
            self._gen_qnn_mirror_pad(call)
        elif call.op == relay.op.get("vision.roi_align"):
            self._gen_roi_align(call)
        elif call.op == relay.op.get("vision.roi_pool"):
            self._gen_roi_pool(call)
        elif call.op == relay.op.get("vision.non_max_suppression"):
            self._gen_non_max_suppression(call)
        elif _is_custom_single_op(call.op):
            func = CODEGEN_CUSTOM_OP_DICT[call.op.name]
            self._gen_custom_op(call, func)
        elif _is_composite_custom_op(call.op):
            func = CODEGEN_CUSTOM_OP_DICT[call.op.attrs["Composite"]]
            self._gen_custom_op(call, func)
        elif call.op == relay.op.get("left_shift"):
            self._gen_left_shift(call)
        elif call.op == relay.op.get("right_shift"):
            self._gen_right_shift(call)
        elif call.op == relay.op.get("trunc"):
            self._gen_trunc(call)
        elif call.op == relay.op.get("bitwise_and"):
            self._gen_bitwise(call, "AND")
        elif call.op == relay.op.get("bitwise_or"):
            self._gen_bitwise(call, "OR")
        elif call.op == relay.op.get("bitwise_xor"):
            self._gen_bitwise(call, "XOR")
        elif call.op == relay.op.get("bitwise_not"):
            self._gen_bitwise(call, "NOT")
        elif call.op == relay.op.get("cumsum"):
            self._gen_cumsum(call)
        elif call.op == relay.op.get("cumprod"):
            self._gen_cumprod(call)
        elif call.op == relay.op.get("meshgrid"):
            self._gen_meshgrid(call)
        elif call.op == relay.op.get("vision.get_valid_counts"):
            self._gen_get_valid_counts(call)
        elif call.op == relay.op.get("vision.multibox_transform_loc"):
            self._gen_multibox_transform_loc(call)
        elif call.op == relay.op.get("contrib.aipu_compass.detection_output"):
            self._gen_detection_output(call)
        elif call.op == relay.op.get("contrib.aipu_compass.nms"):
            self._gen_nms(call)
        elif call.op == relay.op.get("contrib.aipu_compass.decode_box"):
            self._gen_decode_box(call)
        elif call.op == relay.op.get("qnn.dequantize"):
            self._gen_dequantize(call)
        elif call.op == relay.op.get("qnn.quantize"):
            self._gen_quantize(call)
        elif _is_composite_op(call.op, "aipu_compass.DenseAddActivation"):
            self._gen_dense_add_activation(call)
        else:
            assert False, f"Unsupported Call Node:\n{call.op}"

    def _gen_convolution2d(self, call):
        func = call.op
        if call.op == relay.op.get("nn.conv2d"):
            add = None
            conv2d = call
            activation = "NONE"
        elif func.body.op == relay.op.get("nn.conv2d") or func.body.op == relay.op.get(
            "nn.conv2d_transpose"
        ):
            add = None
            conv2d = func.body
            activation = "NONE"
        elif func.body.op == relay.op.get("add"):
            add = func.body
            conv2d, _ = unpack_commutative_args(add)
            activation = "NONE"
        elif func.body.op == relay.op.get("nn.relu"):
            relu = func.body
            if relu.args[0].op == relay.op.get("add"):
                add = relu.args[0]
                conv2d, _ = unpack_commutative_args(add)
            else:
                add = None
                conv2d = relu.args[0]
            activation = "RELU"
        elif func.body.op == relay.op.get("clip"):
            clip = func.body
            add = clip.args[0]
            conv2d, _ = unpack_commutative_args(add)
            activation = "RELU6"
        elif func.body.op == relay.op.get("nn.leaky_relu"):
            leaky_relu = func.body
            if leaky_relu.args[0].op == relay.op.get("add"):
                add = leaky_relu.args[0]
                conv2d, _ = unpack_commutative_args(add)
            else:
                add = None
                conv2d = leaky_relu.args[0]
            activation = "LEAKYRELU"

        input_tensor = conv2d.args[0]
        weight = conv2d.args[1]
        conv2d_attrs = conv2d.attrs
        is_dw_conv = conv2d_attrs.groups == input_tensor.checked_type.shape[-1]
        weight_dtype = weight.checked_type.dtype
        weight_shape = weight.checked_type.shape
        if (
            is_dw_conv
            and weight_shape[-1] != 1
            and conv2d.op != relay.op.get("nn.conv2d_transpose")
        ):
            np_weight = weight.data.numpy()
            np_weight = np_weight.transpose(0, 3, 1, 2)
            np_weight = np_weight.reshape(
                np_weight.shape[0] * np_weight.shape[1], 1, np_weight.shape[2], np_weight.shape[3]
            )
            np_weight = np_weight.transpose(0, 2, 3, 1)
            weight = relay.const(np_weight, weight_dtype)
            weight_shape = list(np_weight.shape)

        if add is not None:
            _, bias = unpack_commutative_args(add)
            bias_dtype = bias.checked_type.dtype
        else:
            zero_bias = np.zeros((1, 1, 1, int(conv2d_attrs.channels)), dtype="float32")
            bias = relay.const(zero_bias)
            bias_dtype = "float32"

        weight_offset, weight_nbytes = self._get_offset_nbytes(weight)
        bias_offset, bias_nbytes = self._get_offset_nbytes(bias)

        pad_bottom = int(conv2d_attrs.padding[2])
        pad_right = int(conv2d_attrs.padding[3])
        if conv2d.op == relay.op.get("nn.conv2d_transpose"):
            layer_type = "ConvTranspose"
            # update pads
            out_pad = conv2d_attrs.output_padding
            pad_bottom -= int(out_pad[0])
            pad_right -= int(out_pad[1])
            if pad_bottom < 0 or pad_right < 0:
                raise RuntimeError("Pads need greater than or equal out_pads.")
        else:
            layer_type = "DepthwiseConv" if is_dw_conv else "Convolution"

        self._gen_basic_layer_items(layer_type, conv2d.args[0], call)
        self._ir_text += textwrap.dedent(
            f"""
            weights_type={weight_dtype}
            weights_offset={weight_offset}
            weights_size={weight_nbytes}
            weights_shape={_verify_shape(weight_shape)}
            biases_type={bias_dtype}
            biases_offset={bias_offset}
            biases_size={bias_nbytes}
            biases_shape={[conv2d_attrs.channels]}
            dilation_x={int(conv2d_attrs.dilation[1])}
            dilation_y={int(conv2d_attrs.dilation[0])}
            group={int(conv2d_attrs.groups)}
            kernel_x={int(conv2d_attrs.kernel_size[1])}
            kernel_y={int(conv2d_attrs.kernel_size[0])}
            num_output={int(conv2d_attrs.channels)}
            pad_bottom={pad_bottom}
            pad_left={int(conv2d_attrs.padding[1])}
            pad_right={pad_right}
            pad_top={int(conv2d_attrs.padding[0])}
            stride_x={int(conv2d_attrs.strides[1])}
            stride_y={int(conv2d_attrs.strides[0])}
            with_activation={activation}"""
        )
        if activation == "LEAKYRELU":
            self._ir_text += textwrap.dedent(
                f"""
                negative_slope_type=float32
                negative_slope_value={leaky_relu.attrs.alpha}"""
            )
        if is_dw_conv:
            self._ir_text += textwrap.dedent(
                f"""
                multiplier={int(conv2d_attrs.channels) // int(conv2d_attrs.groups)}"""
            )
        self._ir_text += textwrap.dedent("\n")

    def _gen_qnn_basiclstm(self, call):
        func = call.op
        func_out = func.body
        assert isinstance(func_out, relay.Tuple), "the output of basiclstm must be a tuple."
        mul_hout, add_cout = func_out
        sigmoid_sp3, tanh_c = mul_hout.args[:2]
        mul_sp2, mul_sp01 = add_cout.args[:2]
        sigmoid_sp0, tanh_sp2 = mul_sp01.args[:2]
        sigmoid_sp1, initial_c = mul_sp2.args[:2]

        split = sigmoid_sp1.args[0].tuple_value
        add_bias = split.args[0]
        add_fc = add_bias.args[0]

        req_after_squeeze = add_fc.args[0]
        q_dense_after_squeeze = req_after_squeeze.args[0]

        # reshape input dim from 2 to 3
        input_x = q_dense_after_squeeze.args[0]
        batch_size, input_size = input_x.checked_type.shape

        input_x_scale = q_dense_after_squeeze.args[4].data.numpy()
        input_x_zp = q_dense_after_squeeze.args[2].data.numpy()

        reshape_id = self._get_layer_idx()
        input_x = _convert_composite_arg(input_x, call)
        input_x_shape = [int(val) for val in input_x.checked_type.shape]
        input_x_type = input_x.checked_type.dtype
        reshape_shape = [batch_size, 1, input_size]
        reshape_out = relay.reshape(input_x, reshape_shape)
        reshape_out_name = self._get_or_alloc_var_name(reshape_out)

        self._ir_text += textwrap.dedent(
            f"""
        layer_id={reshape_id}
        layer_name={reshape_id}_reshape
        layer_type=Reshape
        layer_bottom=[{self._get_or_alloc_var_name(input_x)}]
        layer_bottom_shape=[{_verify_shape(input_x_shape)}]
        layer_bottom_type=[{input_x_type}]
        layer_top=[{reshape_out_name}]
        layer_top_shape=[{reshape_shape}]
        layer_top_type=[{input_x_type}]
        layer_top_scale=[[{input_x_scale}]]
        layer_top_zp=[[{input_x_zp}]]
        shape={reshape_shape}
        """
        )

        # Parameter weight
        weight = q_dense_after_squeeze.args[1]
        weight_dtype = weight.checked_type.dtype
        weight_scale = q_dense_after_squeeze.args[5]
        weight_zp = q_dense_after_squeeze.args[3]

        # Recurrence weight
        req_after_hstate = add_fc.args[1]
        q_dense_after_hstate = req_after_hstate.args[0]
        initial_h, recurrent_weight = q_dense_after_hstate.args[:2]
        _, hidden_size = initial_h.checked_type.shape
        recurrent_weight_scale = q_dense_after_hstate.args[5]
        recurrent_weight_zp = q_dense_after_hstate.args[3]

        # Bias
        bias = add_bias.args[1]
        bias_dtype = bias.checked_type.dtype
        biases_scale, biases_zp = add_bias.args[4:6]

        # Generates weights&bias defined by Compass
        np_weight = weight.data.numpy()
        np_recurrent_weight = recurrent_weight.data.numpy()
        np_bias = np.squeeze(bias.data.numpy())

        i_w, f_w, c_w, o_w = np.split(np_weight, 4, axis=0)
        i_r, f_r, c_r, o_r = np.split(np_recurrent_weight, 4, axis=0)
        i_wb, f_wb, c_wb, o_wb = np.split(np_bias, 4, axis=0)
        i_rb, f_rb, c_rb, o_rb = [np.zeros_like(i) for i in [i_wb, f_wb, c_wb, o_wb]]
        i_weights = np.concatenate([i_w, i_r], axis=1)
        o_weights = np.concatenate([o_w, o_r], axis=1)
        f_weights = np.concatenate([f_w, f_r], axis=1)
        c_weights = np.concatenate([c_w, c_r], axis=1)
        i_biases = i_wb + i_rb
        o_biases = o_wb + o_rb
        f_biases = f_wb + f_rb
        c_biases = c_wb + c_rb
        weights = np.concatenate([i_weights, c_weights, f_weights, o_weights], axis=0)
        biases = np.concatenate([i_biases, c_biases, f_biases, o_biases], axis=0)

        weight_shape = list(weights.shape)
        weights = relay.const(weights, weight_dtype)
        weight_offset, weight_nbytes = self._get_offset_nbytes(weights)
        weights_scale = np.array(
            [weight_scale.data.numpy(), recurrent_weight_scale.data.numpy()], np.float32
        )
        weights_scale_shape = list(weights_scale.shape)
        weights_scale = relay.const(weights_scale, "float32")
        weights_zp = np.array([weight_zp.data.numpy(), recurrent_weight_zp.data.numpy()], np.int32)
        weights_zp_shape = list(weights_zp.shape)
        weights_zp = relay.const(weights_zp, "int32")
        weights_scale_offset, weights_scale_nbytes = self._get_offset_nbytes(weights_scale)
        weights_zp_offset, weights_zp_nbytes = self._get_offset_nbytes(weights_zp)

        biases_shape = list(biases.shape)
        biases = relay.const(biases, bias_dtype)
        bias_offset, bias_nbytes = self._get_offset_nbytes(biases)
        biases_scale_offset, biases_scale_nbytes = self._get_offset_nbytes(biases_scale)
        biases_zp_offset, biases_zp_nbytes = self._get_offset_nbytes(biases_zp)

        # Save activation's scale&zp, which qtlib will use later
        activations_scale, activations_zp = [], []
        for node in [
            add_bias,
            sigmoid_sp0,
            tanh_sp2,
            sigmoid_sp1,
            sigmoid_sp3,
            mul_sp01,
            mul_sp2,
            tanh_c,
            add_cout,
            mul_hout,
        ]:
            assert len(node.args) in [5, 8], "Expect args number of node is 5 or 8"
            if len(node.args) == 5:
                scale, zero_point = node.args[3:5]
            else:
                scale, zero_point = node.args[6:8]
            activations_scale.append(scale.data.numpy())
            activations_zp.append(zero_point.data.numpy())
        np_activations_scale = np.array(activations_scale, np.float32)
        activ_scale_shape = list(np_activations_scale.shape)
        np_activations_zp = np.array(activations_zp, np.int32)
        activ_zp_shape = list(np_activations_zp.shape)
        activ_scale = relay.const(np_activations_scale, "float32")
        activ_zp = relay.const(np_activations_zp, "int32")
        activ_scale_offset, activ_scale_nbytes = self._get_offset_nbytes(activ_scale)
        activ_zp_offset, activ_zp_nbytes = self._get_offset_nbytes(activ_zp)

        layer_type = "BasicLSTM"
        layer_idx = self._get_layer_idx()

        input_names = []
        input_shapes = []
        input_types = []
        input_names.append(reshape_out_name)
        input_shapes.append(reshape_shape)
        input_types.append(input_x_type)

        for arg in [initial_h, initial_c]:
            arg = _convert_composite_arg(arg, call)
            input_names.append(self._get_or_alloc_var_name(arg))
            input_shapes.append(arg.checked_type.shape)
            input_types.append(arg.checked_type.dtype)

        output_names = []
        output_shapes = []
        output_types = []
        output_scales = []
        output_zps = []

        if isinstance(call.checked_type, relay.TupleType):
            for out in self._tuple2tgn[call]:
                output_names.append(self._get_or_alloc_var_name(out))
                qnn_params = self._get_qnn_params(out, is_to_list=True)
                if qnn_params:
                    scale, zero_point = qnn_params
                    output_scales.append(scale)
                    output_zps.append(zero_point)
            for field in call.checked_type.fields:
                output_shapes.append(field.shape)
                output_types.append(field.dtype)

        useless_out = len(output_shapes) - len(output_names)
        for i in range(useless_out):
            output_names.append("useless_out_" + str(self._get_layer_idx()) + str(i))
            if output_scales and output_zps:
                output_scales.append(output_scales[0])
                output_zps.append(output_zps[0])

        self._ir_text += textwrap.dedent(
            f"""
            layer_id={layer_idx}
            layer_name={layer_idx}_{layer_type.lower()}
            layer_type={layer_type}
            layer_bottom=[{", ".join(input_names)}]
            layer_bottom_shape={_verify_shape(input_shapes)}
            layer_bottom_type=[{", ".join(input_types)}]
            layer_top=[{", ".join(output_names)}]
            layer_top_shape={_verify_shape(output_shapes)}
            layer_top_type=[{", ".join(output_types)}]"""
        )
        if output_scales and output_zps:
            assert len(output_scales) == len(output_zps) == len(output_names)
            self._ir_text += textwrap.dedent(
                f"""
                layer_top_scale={output_scales}
                layer_top_zp={output_zps}"""
            )

        self._ir_text += textwrap.dedent(
            f"""
            weights_type={weight_dtype}
            weights_offset={weight_offset}
            weights_size={weight_nbytes}
            weights_shape={_verify_shape(weight_shape)}
            weights_scale_type=float32
            weights_scale_offset={weights_scale_offset}
            weights_scale_size={weights_scale_nbytes}
            weights_scale_shape={weights_scale_shape}
            weights_zp_type=int32
            weights_zp_offset={weights_zp_offset}
            weights_zp_size={weights_zp_nbytes}
            weights_zp_shape={weights_zp_shape}
            biases_type={bias_dtype}
            biases_offset={bias_offset}
            biases_size={bias_nbytes}
            biases_shape={biases_shape}
            biases_scale_type=float32
            biases_scale_offset={biases_scale_offset}
            biases_scale_size={biases_scale_nbytes}
            biases_scale_shape={_verify_shape(biases_scale.checked_type.shape)}
            biases_zp_type=int32
            biases_zp_offset={biases_zp_offset}
            biases_zp_size={biases_zp_nbytes}
            biases_zp_shape={_verify_shape(biases_zp.checked_type.shape)}
            time_steps=1
            input_size={input_size}
            cell_size={hidden_size}
            activations=[SIGMOID,TANH,TANH]
            direction=forward
            activations_scale_type=float32
            activations_scale_offset={activ_scale_offset}
            activations_scale_size={activ_scale_nbytes}
            activations_scale_shape={activ_scale_shape}
            activations_zp_type=int32
            activations_zp_offset={activ_zp_offset}
            activations_zp_size={activ_zp_nbytes}
            activations_zp_shape={activ_zp_shape}
            out_sequence=[H,C]"""
        )
        self._ir_text += textwrap.dedent("\n")

    def _gen_qnn_convolution2d(self, call):
        func = call.op
        if func.body.op == relay.op.get("qnn.requantize"):
            requantize = func.body
            activation = "NONE"
            if requantize.args[0].op == relay.op.get("qnn.leaky_relu"):
                leaky_relu = requantize.args[0]
                requantize = leaky_relu.args[0]
                activation = "LEAKYRELU"
        elif func.body.op == relay.op.get("qnn.leaky_relu"):
            leaky_relu = func.body
            requantize = leaky_relu.args[0]
            activation = "LEAKYRELU"
        elif func.body.op == relay.op.get("clip"):
            clip = func.body
            requantize = clip.args[0]
            activation = get_activation_str(requantize.args[3], requantize.args[4], clip)
        elif func.body.op == relay.op.get("qnn.quantize"):
            quantize = func.body
            if quantize.args[0].op == relay.op.get("nn.leaky_relu"):
                activation = "LEAKYRELU"
                leaky_relu = quantize.args[0]
            dequantize = quantize.args[0].args[0]
            requantize = dequantize.args[0]

        if requantize.args[0].op == relay.op.get("add"):
            add = requantize.args[0]
            qnn_conv2d, _ = unpack_commutative_args(add)
        else:
            add = None
            qnn_conv2d = requantize.args[0]

        input_tensor = qnn_conv2d.args[0]
        weight = qnn_conv2d.args[1]
        qnn_conv2d_attrs = qnn_conv2d.attrs
        in_channel = input_tensor.checked_type.shape[-1]
        is_dw_conv = qnn_conv2d_attrs.groups == in_channel and in_channel != 1

        if add is not None:
            _, bias = unpack_commutative_args(add)
            bias_dtype = bias.checked_type.dtype
        else:
            zero_bias = np.zeros((1, 1, 1, int(qnn_conv2d_attrs.channels)), dtype="int32")
            bias = relay.const(zero_bias)
            bias_dtype = "int32"

        # weight
        weight_dtype = weight.checked_type.dtype
        weight_shape = weight.checked_type.shape
        if (
            is_dw_conv
            and weight_shape[-1] != 1
            and qnn_conv2d.op != relay.op.get("qnn.conv2d_transpose")
        ):
            np_weight = weight.data.numpy()
            np_weight = np_weight.transpose(0, 3, 1, 2)
            np_weight = np_weight.reshape(
                np_weight.shape[0] * np_weight.shape[1], 1, np_weight.shape[2], np_weight.shape[3]
            )
            np_weight = np_weight.transpose(0, 2, 3, 1)
            weight = relay.const(np_weight, weight_dtype)
            weight_shape = list(np_weight.shape)
        weight_offset, weight_nbytes = self._get_offset_nbytes(weight)

        # weight scale & zp
        weights_scale = qnn_conv2d.args[5]
        weights_scale_offset, weights_scale_nbytes = self._get_offset_nbytes(weights_scale)

        weights_zp = qnn_conv2d.args[3]
        weights_zp_offset, weights_zp_nbytes = self._get_offset_nbytes(weights_zp)

        # bias
        bias_offset, bias_nbytes = self._get_offset_nbytes(bias)

        # bias scale & zp
        biases_scale = requantize.args[1]
        biases_scale_offset, biases_scale_nbytes = self._get_offset_nbytes(biases_scale)

        biases_zp = requantize.args[2]
        biases_zp_offset, biases_zp_nbytes = self._get_offset_nbytes(biases_zp)

        # pad
        pad_bottom = int(qnn_conv2d_attrs.padding[2])
        pad_right = int(qnn_conv2d_attrs.padding[3])
        if qnn_conv2d.op == relay.op.get("qnn.conv2d_transpose"):
            layer_type = "ConvTranspose"
            # update pads
            out_pad = qnn_conv2d_attrs.output_padding
            pad_bottom -= int(out_pad[0])
            pad_right -= int(out_pad[1])
            if pad_bottom < 0 or pad_right < 0:
                raise RuntimeError("Pads need greater than or equal out_pads.")
        else:
            layer_type = "DepthwiseConv" if is_dw_conv else "Convolution"

        self._gen_basic_layer_items(layer_type, qnn_conv2d.args[0], call)
        self._ir_text += textwrap.dedent(
            f"""
            weights_type={weight_dtype}
            weights_offset={weight_offset}
            weights_size={weight_nbytes}
            weights_shape={_verify_shape(weight_shape)}
            weights_scale_type=float32
            weights_scale_offset={weights_scale_offset}
            weights_scale_size={weights_scale_nbytes}
            weights_scale_shape={_verify_shape(weights_scale.checked_type.shape)}
            weights_zp_type=int32
            weights_zp_offset={weights_zp_offset}
            weights_zp_size={weights_zp_nbytes}
            weights_zp_shape={_verify_shape(weights_zp.checked_type.shape)}
            biases_type={bias_dtype}
            biases_offset={bias_offset}
            biases_size={bias_nbytes}
            biases_shape={[int(qnn_conv2d_attrs.channels)]}
            biases_scale_type=float32
            biases_scale_offset={biases_scale_offset}
            biases_scale_size={biases_scale_nbytes}
            biases_scale_shape={_verify_shape(biases_scale.checked_type.shape)}
            biases_zp_type=int32
            biases_zp_offset={biases_zp_offset}
            biases_zp_size={biases_zp_nbytes}
            biases_zp_shape={_verify_shape(biases_zp.checked_type.shape)}
            num_output={int(qnn_conv2d_attrs.channels)}
            kernel_x={int(qnn_conv2d_attrs.kernel_size[1])}
            kernel_y={int(qnn_conv2d_attrs.kernel_size[0])}
            stride_x={int(qnn_conv2d_attrs.strides[1])}
            stride_y={int(qnn_conv2d_attrs.strides[0])}
            pad_left={int(qnn_conv2d_attrs.padding[1])}
            pad_right={pad_right}
            pad_top={int(qnn_conv2d_attrs.padding[0])}
            pad_bottom={pad_bottom}
            dilation_x={int(qnn_conv2d_attrs.dilation[1])}
            dilation_y={int(qnn_conv2d_attrs.dilation[0])}
            group={int(qnn_conv2d_attrs.groups)}
            with_activation={activation}"""
        )
        if activation == "LEAKYRELU":
            self._ir_text += textwrap.dedent(
                f"""
                negative_slope_type=float32
                negative_slope_value={leaky_relu.attrs.alpha}"""
            )
        if is_dw_conv:
            self._ir_text += textwrap.dedent(
                f"""
                multiplier={int(qnn_conv2d_attrs.channels) // int(qnn_conv2d_attrs.groups)}"""
            )
        if layer_type == "ConvTranspose":
            self._ir_text += textwrap.dedent(
                """
                output_padding_x=0
                output_padding_y=0"""
            )
        self._ir_text += textwrap.dedent("\n")

    def _gen_convolution3d(self, call):
        func = call.op
        if func.body.op == relay.op.get("nn.conv3d"):
            add = None
            conv3d = func.body
            activation = "NONE"
        if func.body.op == relay.op.get("add"):
            add = func.body
            conv3d, _ = unpack_commutative_args(add)
            activation = "NONE"
        if func.body.op == relay.op.get("nn.relu"):
            relu = func.body
            if relu.args[0].op == relay.op.get("add"):
                add = relu.args[0]
                conv3d, _ = unpack_commutative_args(add)
            else:
                add = None
                conv3d = relu.args[0]
            activation = "RELU"
        if func.body.op == relay.op.get("clip"):
            clip = func.body
            add = clip.args[0]
            conv3d, _ = unpack_commutative_args(add)
            activation = "RELU6"
        if func.body.op == relay.op.get("nn.leaky_relu"):
            leaky_relu = func.body
            if leaky_relu.args[0].op == relay.op.get("add"):
                add = leaky_relu.args[0]
                conv3d, _ = unpack_commutative_args(add)
            else:
                add = None
                conv3d = leaky_relu.args[0]
            activation = "LEAKYRELU"

        weight = conv3d.args[1]
        conv3d_attrs = conv3d.attrs
        weight_dtype = weight.checked_type.dtype
        weight_shape = weight.checked_type.shape

        if add is not None:
            _, bias = unpack_commutative_args(add)
            bias_dtype = bias.checked_type.dtype
        else:
            zero_bias = np.zeros((int(conv3d_attrs.channels),), dtype="float32")
            bias = relay.const(zero_bias)
            bias_dtype = "float32"

        weight_offset, weight_nbytes = self._get_offset_nbytes(weight)
        bias_offset, bias_nbytes = self._get_offset_nbytes(bias)

        self._gen_basic_layer_items("Convolution3D", conv3d.args[0], call)
        self._ir_text += textwrap.dedent(
            f"""
            weights_type={weight_dtype}
            weights_offset={weight_offset}
            weights_size={weight_nbytes}
            weights_shape={_verify_shape(weight_shape)}
            biases_type={bias_dtype}
            biases_offset={bias_offset}
            biases_size={bias_nbytes}
            biases_shape={[conv3d_attrs.channels]}
            dilation_x={int(conv3d_attrs.dilation[2])}
            dilation_y={int(conv3d_attrs.dilation[1])}
            dilation_z={int(conv3d_attrs.dilation[0])}
            group={int(conv3d_attrs.groups)}
            kernel_x={int(conv3d_attrs.kernel_size[2])}
            kernel_y={int(conv3d_attrs.kernel_size[1])}
            kernel_z={int(conv3d_attrs.kernel_size[0])}
            num_output={int(conv3d_attrs.channels)}
            pad_z_begin={int(conv3d_attrs.padding[0])}
            pad_y_begin={int(conv3d_attrs.padding[1])}
            pad_x_begin={int(conv3d_attrs.padding[2])}
            pad_z_end={int(conv3d_attrs.padding[3])}
            pad_y_end={int(conv3d_attrs.padding[4])}
            pad_x_end={int(conv3d_attrs.padding[5])}
            stride_x={int(conv3d_attrs.strides[2])}
            stride_y={int(conv3d_attrs.strides[1])}
            stride_z={int(conv3d_attrs.strides[0])}
            with_activation={activation}"""
        )
        if activation == "LEAKYRELU":
            self._ir_text += textwrap.dedent(
                f"""
                negative_slope_type=float32
                negative_slope_value={leaky_relu.attrs.alpha}
                """
            )
        else:
            self._ir_text += textwrap.dedent("\n")

    def _gen_elementwise_relu(self, call):
        func = call.op
        elt = func.body.args[0]
        elt_op = elt.op
        op_method = {"add": "ADD", "multiply": "MUL", "subtract": "SUB"}

        self._gen_basic_layer_items("Eltwise", elt.args, call)
        self._ir_text += textwrap.dedent(
            f"""
            method={op_method[elt_op.name]}
            with_activation=RELU
            """
        )

    def _gen_qnn_minimum(self, call):
        func = call.op
        req = func.body
        clip = req.args[0]
        minimum = clip.args[0]
        activation = get_activation_str(req.args[1], req.args[2], clip)

        self._gen_basic_layer_items("Eltwise", minimum.args, call)
        self._ir_text += textwrap.dedent(
            f"""
            with_activation={activation}
            method=MIN
            """
        )

    def _gen_qnn_eltwise_relu(self, call):
        func = call.op
        clip = func.body
        qnn_eltwise = clip.args[0]
        activation = get_activation_str(qnn_eltwise.args[6], qnn_eltwise.args[7], clip)

        self._gen_basic_layer_items("Eltwise", qnn_eltwise.args[:2], call)
        self._ir_text += textwrap.dedent(
            f"""
            with_activation={activation}
            method={qnn_eltwise.op.name[4:7].upper()}
            """
        )

    def _gen_dense_add(self, call):
        func = call.op
        add = func.body
        dense, bias = unpack_commutative_args(add)
        weight = dense.args[1]

        weight_offset, weight_nbytes = self._get_offset_nbytes(weight)
        bias_offset, bias_nbytes = self._get_offset_nbytes(bias)
        weight_ttype = weight.checked_type
        bias_ttype = bias.checked_type
        output_shape = _verify_shape(call.checked_type.shape)

        self._gen_basic_layer_items("FullyConnected", dense.args[0], call)
        self._ir_text += textwrap.dedent(
            f"""
            weights_type={weight_ttype.dtype}
            weights_offset={weight_offset}
            weights_size={weight_nbytes}
            weights_shape={_verify_shape(weight_ttype.shape)}
            biases_type={bias_ttype.dtype}
            biases_offset={bias_offset}
            biases_size={bias_nbytes}
            biases_shape={[int(x) for x in bias_ttype.shape if x != 1] or [1]}
            num_output={output_shape[-1]}
            with_activation=NONE
            """
        )

    def _gen_qnn_dense_add(self, call):
        func = call.op
        requantize = func.body
        qnn_dense, bias = unpack_commutative_args(requantize.args[0])
        weight = qnn_dense.args[1]

        weight_offset, weight_nbytes = self._get_offset_nbytes(weight)
        bias_offset, bias_nbytes = self._get_offset_nbytes(bias)
        weight_ttype = weight.checked_type
        bias_ttype = bias.checked_type

        # weight scale & zp
        weights_scale = qnn_dense.args[5]
        weights_scale_offset, weights_scale_nbytes = self._get_offset_nbytes(weights_scale)

        weights_zp = qnn_dense.args[3]
        weights_zp_offset, weights_zp_nbytes = self._get_offset_nbytes(weights_zp)

        # bias scale & zp
        biases_scale = requantize.args[1]
        biases_scale_offset, biases_scale_nbytes = self._get_offset_nbytes(biases_scale)

        biases_zp = requantize.args[2]
        biases_zp_offset, biases_zp_nbytes = self._get_offset_nbytes(biases_zp)

        output_shape = _verify_shape(call.checked_type.shape)

        self._gen_basic_layer_items("FullyConnected", qnn_dense.args[0], call)
        self._ir_text += textwrap.dedent(
            f"""
            weights_type={weight_ttype.dtype}
            weights_offset={weight_offset}
            weights_size={weight_nbytes}
            weights_shape={_verify_shape(weight_ttype.shape)}
            weights_scale_type=float32
            weights_scale_offset={weights_scale_offset}
            weights_scale_size={weights_scale_nbytes}
            weights_scale_shape={_verify_shape(weights_scale.checked_type.shape)}
            weights_zp_type=int32
            weights_zp_offset={weights_zp_offset}
            weights_zp_size={weights_zp_nbytes}
            weights_zp_shape={_verify_shape(weights_zp.checked_type.shape)}
            biases_type={bias_ttype.dtype}
            biases_offset={bias_offset}
            biases_size={bias_nbytes}
            biases_shape={[int(x) for x in bias_ttype.shape if x != 1] or [1]}
            biases_scale_type=float32
            biases_scale_offset={biases_scale_offset}
            biases_scale_size={biases_scale_nbytes}
            biases_scale_shape={_verify_shape(biases_scale.checked_type.shape)}
            biases_zp_type=int32
            biases_zp_offset={biases_zp_offset}
            biases_zp_size={biases_zp_nbytes}
            biases_zp_shape={_verify_shape(biases_zp.checked_type.shape)}
            num_output={int(output_shape[-1])}
            with_activation=NONE
            """
        )

    def _gen_dense(self, call):
        func = call.op
        dense = func.body
        add_shape = dense.checked_type.shape[-1]
        add_shape = [int(add_shape)]
        dtype = dense.checked_type.dtype
        add_zero = np.zeros(add_shape, dtype=dtype)
        add_zero = relay.const(add_zero)
        add_zero._checked_type_ = ir.TensorType(add_shape, dtype=dtype)
        weight = dense.args[1]

        weight_offset, weight_nbytes = self._get_offset_nbytes(weight)
        bias_offset, bias_nbytes = self._get_offset_nbytes(add_zero)
        weight_ttype = weight.checked_type
        bias_ttype = add_zero.checked_type
        output_shape = _verify_shape(call.checked_type.shape)

        self._gen_basic_layer_items("FullyConnected", dense.args[0], call)
        self._ir_text += textwrap.dedent(
            f"""
            weights_type={weight_ttype.dtype}
            weights_offset={weight_offset}
            weights_size={weight_nbytes}
            weights_shape={_verify_shape(weight_ttype.shape)}
            biases_type={bias_ttype.dtype}
            biases_offset={bias_offset}
            biases_size={bias_nbytes}
            biases_shape={[int(x) for x in bias_ttype.shape if x != 1] or [1]}
            num_output={output_shape[-1]}
            with_activation=NONE
            """
        )

    def _gen_qnn_dense(self, call):
        func = call.op
        if func.body.op == relay.op.get("qnn.requantize"):
            requantize = func.body
            activation = "NONE"
        elif func.body.op == relay.op.get("clip"):
            clip = func.body
            requantize = clip.args[0]
            activation = get_activation_str(requantize.args[3], requantize.args[4], clip)

        qnn_dense = requantize.args[0]
        add_shape = qnn_dense.checked_type.shape[-1]
        add_shape = [int(add_shape)]
        dtype = qnn_dense.checked_type.dtype
        add_zero = np.zeros(add_shape, dtype=dtype)
        bias = relay.const(add_zero)
        bias._checked_type_ = ir.TensorType(add_shape, dtype=dtype)
        weight = qnn_dense.args[1]

        weight_offset, weight_nbytes = self._get_offset_nbytes(weight)
        bias_offset, bias_nbytes = self._get_offset_nbytes(bias)
        weight_ttype = weight.checked_type
        bias_ttype = bias.checked_type

        # weight scale & zp
        weights_scale = qnn_dense.args[5]
        weights_scale_offset, weights_scale_nbytes = self._get_offset_nbytes(weights_scale)

        weights_zp = qnn_dense.args[3]
        weights_zp_offset, weights_zp_nbytes = self._get_offset_nbytes(weights_zp)

        # bias scale & zp
        biases_scale = requantize.args[1]
        biases_scale_offset, biases_scale_nbytes = self._get_offset_nbytes(biases_scale)

        biases_zp = requantize.args[2]
        biases_zp_offset, biases_zp_nbytes = self._get_offset_nbytes(biases_zp)

        output_shape = _verify_shape(call.checked_type.shape)

        self._gen_basic_layer_items("FullyConnected", qnn_dense.args[0], call)
        self._ir_text += textwrap.dedent(
            f"""
            weights_type={weight_ttype.dtype}
            weights_offset={weight_offset}
            weights_size={weight_nbytes}
            weights_shape={_verify_shape(weight_ttype.shape)}
            weights_scale_type=float32
            weights_scale_offset={weights_scale_offset}
            weights_scale_size={weights_scale_nbytes}
            weights_scale_shape={_verify_shape(weights_scale.checked_type.shape)}
            weights_zp_type=int32
            weights_zp_offset={weights_zp_offset}
            weights_zp_size={weights_zp_nbytes}
            weights_zp_shape={_verify_shape(weights_zp.checked_type.shape)}
            biases_type={bias_ttype.dtype}
            biases_offset={bias_offset}
            biases_size={bias_nbytes}
            biases_shape={[int(x) for x in bias_ttype.shape if x != 1] or [1]}
            biases_scale_type=float32
            biases_scale_offset={biases_scale_offset}
            biases_scale_size={biases_scale_nbytes}
            biases_scale_shape={_verify_shape(biases_scale.checked_type.shape)}
            biases_zp_type=int32
            biases_zp_offset={biases_zp_offset}
            biases_zp_size={biases_zp_nbytes}
            biases_zp_shape={_verify_shape(biases_zp.checked_type.shape)}
            num_output={output_shape[-1]}
            with_activation={activation}
            """
        )

    def _gen_qnn_matmul(self, call):
        expand_dims = call.op.body
        requantize = expand_dims.args[0]
        qdense = requantize.args[0]
        reshape0 = qdense.args[0]
        transpose = qdense.args[1]
        reshape1 = transpose.args[0]
        input0 = reshape0.args[0]
        input1 = reshape1.args[0]

        self._gen_basic_layer_items("MatMul", [input0, input1], call)
        self._ir_text += textwrap.dedent(
            """
            trans_a=false
            trans_b=false
            """
        )

    def _gen_elementwise(self, call):
        self._gen_basic_layer_items("Eltwise", call.args, call)
        self._ir_text += textwrap.dedent(
            f"""
            with_activation=NONE
            method={call.op.name[:3].upper()}
            """
        )

    def _gen_qnn_elementwise(self, call):
        self._gen_basic_layer_items("Eltwise", call.args[:2], call)
        self._ir_text += textwrap.dedent(
            f"""
            with_activation=NONE
            method={call.op.name[4:7].upper()}
            """
        )

    def _gen_divide(self, divide):
        self._gen_basic_layer_items("Div", divide.args, divide)
        self._ir_text += textwrap.dedent("\n")

    def _gen_divide_mod(self, divide_mod):
        self._gen_basic_layer_items("DivMod", divide_mod.args, divide_mod)
        self._ir_text += textwrap.dedent("\n")

    def _gen_mod(self, mod):
        self._gen_basic_layer_items("Mod", mod.args, mod)
        # The relay equivalent of np.fmod is relay.mod, so set fmod as true
        self._ir_text += textwrap.dedent(
            """
            fmod=true
            """
        )

    def _gen_max_pooling2d(self, max_pool2d):
        attrs = max_pool2d.attrs

        self._gen_basic_layer_items("Pooling", max_pool2d.args, max_pool2d)
        self._ir_text += textwrap.dedent(
            f"""
            ceil_mode={bool(attrs.ceil_mode)}
            dilation_x={int(attrs.dilation[1])}
            dilation_y={int(attrs.dilation[0])}
            kernel_x={int(attrs.pool_size[1])}
            kernel_y={int(attrs.pool_size[0])}
            method=MAX
            pad_bottom={int(attrs.padding[2])}
            pad_left={int(attrs.padding[1])}
            pad_right={int(attrs.padding[3])}
            pad_top={int(attrs.padding[0])}
            stride_x={int(attrs.strides[1])}
            stride_y={int(attrs.strides[0])}
            """
        )

    def _gen_avg_pooling2d(self, avg_pool2d):
        attrs = avg_pool2d.attrs

        self._gen_basic_layer_items("Pooling", avg_pool2d.args, avg_pool2d)
        self._ir_text += textwrap.dedent(
            f"""
            ceil_mode={bool(attrs.ceil_mode)}
            count_include_pad={bool(attrs.count_include_pad)}
            dilation_x={int(attrs.dilation[1])}
            dilation_y={int(attrs.dilation[0])}
            kernel_x={int(attrs.pool_size[1])}
            kernel_y={int(attrs.pool_size[0])}
            method=AVG
            pad_bottom={int(attrs.padding[2])}
            pad_left={int(attrs.padding[1])}
            pad_right={int(attrs.padding[3])}
            pad_top={int(attrs.padding[0])}
            stride_x={int(attrs.strides[1])}
            stride_y={int(attrs.strides[0])}
            """
        )

    def _gen_qnn_avg_pool2d(self, call):
        func = call.op
        out_cast = func.body
        avg_pool2d = out_cast.args[0]
        in_cast = avg_pool2d.args[0]

        attrs = avg_pool2d.attrs

        self._gen_basic_layer_items("Pooling", in_cast.args, call)
        self._ir_text += textwrap.dedent(
            f"""
            ceil_mode={bool(attrs.ceil_mode)}
            count_include_pad={bool(attrs.count_include_pad)}
            dilation_x={int(attrs.dilation[1])}
            dilation_y={int(attrs.dilation[0])}
            kernel_x={int(attrs.pool_size[1])}
            kernel_y={int(attrs.pool_size[0])}
            method=AVG
            pad_bottom={int(attrs.padding[2])}
            pad_left={int(attrs.padding[1])}
            pad_right={int(attrs.padding[3])}
            pad_top={int(attrs.padding[0])}
            stride_x={int(attrs.strides[1])}
            stride_y={int(attrs.strides[0])}
            """
        )

    def _gen_global_pooling2d(self, global_pool2d_node, method):
        inputs = global_pool2d_node.args[0]
        inputs_shape = inputs.checked_type.shape

        self._gen_basic_layer_items("Pooling", global_pool2d_node.args, global_pool2d_node)
        self._ir_text += textwrap.dedent(
            f"""
            ceil_mode=True
            count_include_pad=True
            dilation_x=1
            dilation_y=1
            kernel_x={int(inputs_shape[2])}
            kernel_y={int(inputs_shape[1])}
            method={method}
            pad_bottom=0
            pad_left=0
            pad_right=0
            pad_top=0
            stride_x=1
            stride_y=1
            """
        )

    def _gen_reduce(self, call):
        data = call.args[0]

        attrs = call.attrs
        dims = len(data.checked_type.shape)
        axis = [int(x) for x in attrs.axis] if attrs.axis else list(range(dims))
        if attrs.exclude:
            axis = [x for x in range(dims) if x not in axis]

        method = call.op.name.upper()
        if method == "VARIANCE" and attrs.unbiased:
            method = "UNBIASED_VARIANCE"

        self._gen_basic_layer_items("Reduce", data, call)
        self._ir_text += textwrap.dedent(
            f"""
            axis={axis}
            method={method}
            """
        )

    def _gen_qnn_reduce(self, call):
        func = call.op
        requantize = func.body
        reduce_call = requantize.args[0]
        data = reduce_call.args[0].args[0]

        attrs = reduce_call.attrs
        dims = len(data.checked_type.shape)
        axis = [int(x) for x in attrs.axis] if attrs.axis else list(range(dims))
        if attrs.exclude:
            axis = [x for x in range(dims) if x not in axis]

        self._gen_basic_layer_items("Reduce", data, call)
        self._ir_text += textwrap.dedent(
            f"""
            axis={axis}
            method={reduce_call.op.name.upper()}
            """
        )

    def _gen_concatenate(self, concat):
        attrs = concat.attrs

        self._gen_basic_layer_items("Concat", concat.args, concat)
        self._ir_text += textwrap.dedent(
            f"""
            axis={attrs.axis}
            """
        )

    def _gen_qnn_concatenate(self, concat):
        attrs = concat.attrs

        self._gen_basic_layer_items("Concat", concat.args[0], concat)
        self._ir_text += textwrap.dedent(
            f"""
            axis={attrs.axis}
            """
        )

    def _gen_reshape(self, reshape):
        out_shape = _verify_shape(reshape.checked_type.shape)
        self._gen_basic_layer_items("Reshape", reshape.args, reshape)
        self._ir_text += textwrap.dedent(
            f"""
            shape={out_shape}
            """
        )

    def _gen_softmax(self, softmax):
        axis = softmax.attrs.axis
        args = softmax.args

        self._gen_basic_layer_items("Softmax", args, softmax)
        self._ir_text += textwrap.dedent(
            f"""
            axis={axis}
            """
        )

    def _gen_qnn_softmax(self, call):
        func = call.op
        quantize = func.body
        softmax = quantize.args[0]
        dequantize = softmax.args[0]

        self._gen_basic_layer_items("Softmax", dequantize.args[0], call)
        self._ir_text += textwrap.dedent(
            f"""
            axis={softmax.attrs.axis}
            """
        )

    def _gen_log_softmax(self, log_softmax):
        if isinstance(log_softmax.op, relay.Function):
            softmax = log_softmax.op.body.args[0]
            axis = softmax.attrs.axis
            args = softmax.args
        else:
            axis = log_softmax.attrs.axis
            args = log_softmax.args

        self._gen_basic_layer_items("LogSoftmax", args, log_softmax)
        self._ir_text += textwrap.dedent(
            f"""
            axis={axis}
            """
        )

    def _gen_batchnorm(self, batchnorm):
        func = batchnorm.op

        add_arg, add_const = unpack_commutative_args(func.body)
        c = int(add_arg.checked_type.shape[-1])
        if not isinstance(add_arg, relay.expr.Var):
            _, mul_const = unpack_commutative_args(add_arg)
        else:
            if func.body.op.name == "add" or func.body.op.name == "subtract":
                mul_const = np.ones([c], dtype=np.float32)
                mul_const = nd.array(mul_const)
                mul_const = relay.Constant(mul_const)
                mul_const._checked_type_ = ir.TensorType([c])
                if func.body.op.name == "subtract":
                    checked_type = add_const._checked_type_
                    np_add_const = add_const.data.numpy()
                    add_const = relay.const(-np_add_const)
                    add_const._checked_type_ = checked_type
            else:
                mul_const = add_const
                add_const = np.zeros([c], dtype=np.float32)
                add_const = nd.array(add_const)
                add_const = relay.Constant(add_const)
                add_const._checked_type_ = ir.TensorType([c])
                if func.body.op.name == "divide":
                    mul_const_shape = mul_const.checked_type.shape
                    np_mul_const = mul_const.data.numpy()
                    mul_const = relay.const(1 / np_mul_const)
                    mul_const._checked_type_ = ir.TensorType(mul_const_shape)

        add_const_shape = add_const.checked_type.shape
        mul_const_shape = mul_const.checked_type.shape
        if len(add_const_shape) == 0 or add_const_shape[-1] == 1:
            add_const = add_const.data.numpy().reshape([-1])
            add_const = np.tile(add_const, [c])
            add_const = nd.array(add_const)
            add_const = relay.Constant(add_const)
            add_const._checked_type_ = ir.TensorType([c])

        if len(mul_const_shape) == 0 or mul_const_shape[-1] == 1:
            mul_const = mul_const.data.numpy().reshape([-1])
            mul_const = np.tile(mul_const, [c])
            mul_const = nd.array(mul_const)
            mul_const = relay.Constant(mul_const)
            mul_const._checked_type_ = ir.TensorType([c])

        weight_offset, weight_nbytes = self._get_offset_nbytes(mul_const)
        bias_offset, bias_nbytes = self._get_offset_nbytes(add_const)
        weight_dtype = mul_const.checked_type.dtype
        weight_shape = mul_const.checked_type.shape[-1:]
        bias_dtype = add_const.checked_type.dtype
        bias_shape = add_const.checked_type.shape[-1:]
        self._gen_basic_layer_items("BatchNorm", func.params, batchnorm)
        dims = len(batchnorm.args[0].checked_type.shape)
        self._ir_text += textwrap.dedent(
            f"""
            weights_type={weight_dtype}
            weights_offset={weight_offset}
            weights_size={weight_nbytes}
            weights_shape={_verify_shape(weight_shape)}
            biases_type={bias_dtype}
            biases_offset={bias_offset}
            biases_size={bias_nbytes}
            biases_shape={_verify_shape(bias_shape)}
            axis={dims - 1}
            """
        )

    def _gen_instancenorm(self, instancenorm):
        func = instancenorm.op
        add = func.body
        add_arg, add_const = unpack_commutative_args(add)
        mul_rsqrt, mul_const = unpack_commutative_args(add_arg)
        _, rsqrt = unpack_commutative_args(mul_rsqrt, "rsqrt")
        add_epsilon = rsqrt.args[0]
        _, epsilon = unpack_commutative_args(add_epsilon)
        epsilon = epsilon.data.numpy()
        weight_offset, weight_nbytes = self._get_offset_nbytes(mul_const)
        bias_offset, bias_nbytes = self._get_offset_nbytes(add_const)
        weight_dtype = mul_const.checked_type.dtype
        weight_shape = mul_const.checked_type.shape[-1:]
        bias_dtype = add_const.checked_type.dtype
        bias_shape = add_const.checked_type.shape[-1:]
        self._gen_basic_layer_items("InstanceNorm", func.params, instancenorm)
        self._ir_text += textwrap.dedent(
            f"""
            weights_type={weight_dtype}
            weights_offset={weight_offset}
            weights_size={weight_nbytes}
            weights_shape={_verify_shape(weight_shape)}
            biases_type={bias_dtype}
            biases_offset={bias_offset}
            biases_size={bias_nbytes}
            biases_shape={_verify_shape(bias_shape)}
            epsilon={epsilon}
            """
        )

    def _gen_l2norm(self, l2norm):
        func = l2norm.op
        divide = func.body
        inp, add_or_maximum = divide.args
        sqrt, eps = unpack_commutative_args(add_or_maximum)
        reduce_sum = sqrt.args[0]

        dim = len(inp.checked_type.shape)
        axis = reduce_sum.attrs.axis
        axis = [x if x >= 0 else x + dim for x in axis]
        epsilon = eps.data.numpy()
        assert epsilon.size == 1

        self._gen_basic_layer_items("Normalization", func.params, l2norm)
        self._ir_text += textwrap.dedent(
            f"""
            axis={axis}
            method=L2
            epsilon={epsilon.reshape([])}
            """
        )

    def _gen_transpose(self, transpose):
        attrs = transpose.attrs

        self._gen_basic_layer_items("Transpose", transpose.args, transpose)
        axes = attrs.axes
        if axes is None:
            axes = list(range(len(transpose.args[0].checked_type.shape)))[::-1]
        self._ir_text += textwrap.dedent(
            f"""
            perm={axes}
            """
        )

    def _gen_copy(self, copy):
        in_shape = _verify_shape(copy.args[0].checked_type.shape)
        axes = list(range(len(in_shape)))

        self._gen_basic_layer_items("Transpose", copy.args, copy)
        self._ir_text += textwrap.dedent(
            f"""
            perm={axes}
            """
        )

    def _gen_one_hot(self, one_hot):
        attrs = one_hot.attrs
        on_value = one_hot.args[1]
        off_value = one_hot.args[2]
        on_value = int(on_value.data.numpy().item())
        off_value = int(off_value.data.numpy().item())

        self._gen_basic_layer_items("OneHot", one_hot.args[0], one_hot)
        self._ir_text += textwrap.dedent(
            f"""
            axis={attrs.axis}
            depth={attrs.depth}
            values={[off_value, on_value]}
            """
        )

    def _gen_exp(self, exp):
        self._gen_basic_layer_items("Exp", exp.args, exp)
        self._ir_text += textwrap.dedent(
            """
            """
        )

    def _gen_log(self, log):
        self._gen_basic_layer_items("Log", log.args, log)
        self._ir_text += textwrap.dedent(
            """
            """
        )

    def _gen_abs(self, abs_node):
        self._gen_basic_layer_items("Abs", abs_node.args, abs_node)
        self._ir_text += textwrap.dedent(
            """
            """
        )

    def _gen_cos(self, cos):
        self._gen_basic_layer_items("Cosine", cos.args, cos)
        self._ir_text += textwrap.dedent(
            """
            """
        )

    def _gen_tile(self, tile):
        attrs = tile.attrs
        reps = _verify_shape(attrs.reps)

        self._gen_basic_layer_items("Tile", tile.args, tile)
        self._ir_text += textwrap.dedent(
            f"""
            repeats={reps}
            """
        )

    def _gen_strided_slice(self, strided_slice):
        attrs = strided_slice.attrs
        input_shape = _verify_shape(strided_slice.type_args[0].shape)

        def clamp(value, idx):
            max_value = input_shape[idx]
            min_value = -input_shape[idx] - 1
            return max(min(int(value), max_value), min_value)

        axes = attrs.axes
        begin = attrs.begin
        end = attrs.end
        strides = attrs.strides
        slice_mode = attrs.slice_mode
        dims = len(input_shape)
        new_begin = [0] * dims
        new_end = [np.iinfo(np.int32).max] * dims
        new_stride = [1] * dims

        if axes is not None:
            for i, axis in enumerate(axes):
                new_begin[int(axis)] = begin[i]
                new_end[int(axis)] = end[i]
                new_stride[int(axis)] = strides[i]
        # for tvm cloud support like begin=[0,0,0,0], end=[x,x,x], stide=[1]
        # without axes.
        else:
            for i in range(dims):
                if i < len(begin):
                    new_begin[i] = begin[i]
                if i < len(end):
                    new_end[i] = end[i]
                if i < len(strides):
                    new_stride[i] = strides[i]

        begin = [clamp(val, idx) for idx, val in enumerate(new_begin)]
        end = [clamp(val, idx) for idx, val in enumerate(new_end)]
        strides = [int(val) for val in new_stride]
        # convert mode 'size' to mode 'end'
        if slice_mode == "size":
            strides = [1] * dims
            for i in range(dims):
                if end[i] == -1:
                    end[i] = input_shape[i]
                else:
                    end[i] += begin[i]
                    end[i] = clamp(end[i], i)

        self._gen_basic_layer_items("Slice", strided_slice.args, strided_slice)
        self._ir_text += textwrap.dedent(
            f"""
            begin={begin}
            end={end}
            strides={strides}
            """
        )

    def _gen_resize(self, resize):
        out_shape = _verify_shape(resize.checked_type.shape)
        attrs = resize.attrs
        method = "nearest"
        if attrs.method != "nearest_neighbor":
            method = "bilinear"

        self._gen_basic_layer_items("Resize", resize.args, resize)
        self._ir_text += textwrap.dedent(
            f"""
            method={method.upper()}
            size={out_shape}
            mode={attrs.coordinate_transformation_mode.upper()}"""
        )
        round_method = attrs.rounding_method.upper()
        if method == "nearest" and round_method:
            self._ir_text += textwrap.dedent(
                f"""
                nearest_mode={round_method}
                """
            )
        else:
            self._ir_text += "\n"

    def _gen_qnn_resize(self, call):
        func = call.op
        quantize = func.body
        resize = quantize.args[0]
        dequantize = resize.args[0]

        out_shape = _verify_shape(resize.checked_type.shape)
        attrs = resize.attrs
        method = "nearest"
        if attrs.method != "nearest_neighbor":
            method = "bilinear"

        self._gen_basic_layer_items("Resize", dequantize.args[0], call)
        self._ir_text += textwrap.dedent(
            f"""
            method={method.upper()}
            size={out_shape}
            mode={attrs.coordinate_transformation_mode.upper()}"""
        )
        round_method = attrs.rounding_method.upper()
        if method == "nearest" and round_method:
            self._ir_text += textwrap.dedent(
                f"""
                nearest_mode={round_method}
                """
            )
        else:
            self._ir_text += "\n"

    def _gen_grid_sample(self, grid_sample):
        attrs = grid_sample.attrs

        self._gen_basic_layer_items("GridSample", grid_sample.args, grid_sample)
        self._ir_text += textwrap.dedent(
            f"""
            method={attrs.method}
            align_corners={bool(attrs.align_corners)}
            padding_mode={attrs.padding_mode}
            """
        )

    def _gen_split(self, split):
        attrs = split.attrs
        axis = attrs.axis
        splits = []
        for field in split.checked_type.fields:
            splits.append(field.shape[axis])

        self._gen_basic_layer_items("Split", split.args, split)
        self._ir_text += textwrap.dedent(
            f"""
            splits={splits}
            axis={attrs.axis}
            """
        )

    def _gen_sigmoid(self, sigmoid):
        self._gen_basic_layer_items("Activation", sigmoid.args, sigmoid)
        self._ir_text += textwrap.dedent(
            """
            method=SIGMOID
            """
        )

    def _gen_qnn_sigmoid_req_q(self, call):
        func = call.op
        if func.body.op == relay.op.get("qnn.requantize"):
            requantize = func.body
            sigmoid = requantize.args[0]
            self._gen_basic_layer_items("Activation", sigmoid.args[0], call)
            self._ir_text += textwrap.dedent(
                """
                method=SIGMOID
                """
            )
        elif func.body.op == relay.op.get("qnn.quantize"):
            quantize = func.body
            sigmoid = quantize.args[0]
            dequantize = sigmoid.args[0]

            self._gen_basic_layer_items("Activation", dequantize.args[0], call)
            self._ir_text += textwrap.dedent(
                """
                method=SIGMOID
                """
            )

    def _gen_qnn_sigmoid(self, sigmoid):
        self._gen_basic_layer_items("Activation", sigmoid.args[0], sigmoid)
        self._ir_text += textwrap.dedent(
            """
            method=SIGMOID
            """
        )

    def _gen_qnn_silu(self, call):
        func = call.op
        composite = func.body
        inp = composite.args[0]

        self._gen_basic_layer_items("Activation", inp, call)
        self._ir_text += textwrap.dedent(
            """
            method=SILU
            """
        )

    def _gen_tanh(self, tanh):
        self._gen_basic_layer_items("Activation", tanh.args, tanh)
        self._ir_text += textwrap.dedent(
            """
            method=TANH
            """
        )

    def _gen_qnn_tanh(self, tanh):
        self._gen_basic_layer_items("Activation", tanh.args[0], tanh)
        self._ir_text += textwrap.dedent(
            """
            method=TANH
            """
        )

    def _gen_qnn_squared_difference(self, call):
        func = call.op
        squared_difference = func.body
        difference = squared_difference.args[0]

        self._gen_basic_layer_items("SquaredDifference", difference.args[:2], call)
        self._ir_text += textwrap.dedent(
            """
            """
        )

    def _gen_pad(self, pad):
        attrs = pad.attrs
        pad_mode = attrs.pad_mode.upper()
        pad_width = attrs.pad_width
        flatten_pad_width = [y for x in pad_width for y in x]

        # If all of the pad width are 0, a "Pad" should be generated.
        if any(x < 0 for x in flatten_pad_width) and not any(x > 0 for x in flatten_pad_width):
            crops = []
            input_shape = pad.args[0].checked_type.shape
            for i, size in enumerate(input_shape):
                begin_idx = 0 - pad_width[i][0]
                end_idx = size + pad_width[i][1]
                crops.append([begin_idx, end_idx])

            self._gen_basic_layer_items("Crop", pad.args[0], pad)
            self._ir_text += textwrap.dedent(
                f"""
                crops={[[int(y) for y in x] for x in crops]}
                """
            )
            return

        constant_value = float(pad.args[1].data.numpy())
        qnn_params = self._get_qnn_params(pad)
        if qnn_params:
            zerop = float(qnn_params[1].data.numpy())
            constant_value -= zerop

        self._gen_basic_layer_items("Pad", pad.args[0], pad)
        self._ir_text += textwrap.dedent(
            f"""
            constant_value={constant_value}
            mode={pad_mode}
            pads={[[int(y) for y in x] for x in attrs.pad_width]}
            """
        )

    def _gen_relu(self, relu):
        self._gen_basic_layer_items("Activation", relu.args, relu)
        self._ir_text += textwrap.dedent(
            """
            method=RELU
            """
        )

    def _gen_prelu(self, prelu):
        negative_slope = prelu.args[1]
        negative_slope_shape = negative_slope.checked_type.shape
        negative_slope_offset, negative_slope_nbytes = self._get_offset_nbytes(negative_slope)
        self._gen_basic_layer_items("Activation", prelu.args[0], prelu)
        self._ir_text += textwrap.dedent(
            f"""
            method=PRELU
            negative_slope_type=float32
            negative_slope_shape={_verify_shape(negative_slope_shape)}
            negative_slope_offset={negative_slope_offset}
            negative_slope_size={negative_slope_nbytes}
            """
        )

    def _gen_qnn_flatten_prelu(self, call):
        func = call.op
        reshape1 = func.body
        qnn_prelu = reshape1.args[0]
        reshape0 = qnn_prelu.args[0]

        negative_slope = qnn_prelu.args[1]
        negative_slope_dtype = negative_slope.checked_type.dtype
        negative_slope_scale = qnn_prelu.args[4].data.numpy()
        negative_slope_zp = qnn_prelu.args[5].data.numpy()
        negative_slope_offset, negative_slope_nbytes = self._get_offset_nbytes(negative_slope)
        self._gen_basic_layer_items("Activation", reshape0.args[0], call)
        self._ir_text += textwrap.dedent(
            f"""
            method=PRELU
            negative_slope_type={negative_slope_dtype}
            negative_slope_shape={_verify_shape(reshape1.checked_type.shape)}
            negative_slope_offset={negative_slope_offset}
            negative_slope_size={negative_slope_nbytes}
            negative_slope_scale=[{negative_slope_scale}]
            negative_slope_zp=[{negative_slope_zp}]
            """
        )

    def _gen_leaky_relu(self, leaky_relu):
        self._gen_basic_layer_items("Activation", leaky_relu.args, leaky_relu)
        self._ir_text += textwrap.dedent(
            f"""
            method=LEAKYRELU
            negative_slope_type=float32
            negative_slope_value={leaky_relu.attrs.alpha}
            """
        )

    def _gen_clip(self, clip):
        attrs = clip.attrs

        self._gen_basic_layer_items("Activation", clip.args, clip)
        self._ir_text += textwrap.dedent(
            f"""
            method=CLIP
            clip_min={attrs.a_min}
            clip_max={attrs.a_max}
            """
        )

    def _gen_cast(self, cast):
        attrs = cast.attrs
        ignore_scale_zp = "true" if cast.args[0].checked_type.dtype == "int32" else "false"
        self._gen_basic_layer_items("Cast", cast.args, cast)
        dtype_mapping = {
            "bool": "uint8",
            "int64": "int32",
        }
        dtype = dtype_mapping[attrs.dtype] if attrs.dtype in dtype_mapping else attrs.dtype
        self._ir_text += textwrap.dedent(
            f"""
            to_dtype={dtype}
            ignore_scale_zp={ignore_scale_zp}
            clip_mode=SATURATION
            """
        )

    def _gen_qnn_cast(self, call):
        ignore_scale_zp = "true" if call.args[0].checked_type.dtype == "int32" else "false"
        func = call.op
        if func == relay.op.get("qnn.requantize"):
            self._gen_basic_layer_items("Cast", call.args[0], call)
            self._ir_text += textwrap.dedent(
                f"""
                to_dtype={call.attrs.out_dtype}
                ignore_scale_zp={ignore_scale_zp}
                clip_mode=SATURATION
                """
            )
            return

        quantize = func.body
        cast = quantize.args[0]
        self._gen_basic_layer_items("Cast", cast.args, call)
        dtype_mapping = {
            "bool": "uint8",
            "int64": "int32",
        }
        dtype = (
            dtype_mapping[quantize.attrs.out_dtype]
            if quantize.attrs.out_dtype in dtype_mapping
            else quantize.attrs.out_dtype
        )
        self._ir_text += textwrap.dedent(
            f"""
            to_dtype={dtype}
            ignore_scale_zp={ignore_scale_zp}
            clip_mode=SATURATION
            """
        )

    def _gen_lrn(self, lrn):
        attrs = lrn.attrs

        if attrs.axis == 1 or attrs.axis == 2:
            method = "WITHIN_CHANNEL"
        else:
            method = "ACROSS_CHANNELS"

        self._gen_basic_layer_items("LRN", lrn.args, lrn)
        self._ir_text += textwrap.dedent(
            f"""
            method={method}
            alpha={attrs.alpha}
            beta={attrs.beta}
            bias={attrs.bias}
            size={attrs.size}
            """
        )

    def _gen_channel_shuffle(self, channel_shuffle):
        self._gen_basic_layer_items("ChannelShuffle", channel_shuffle.args[0], channel_shuffle)
        self._ir_text += textwrap.dedent(
            f"""
            splits=1
            group={channel_shuffle.attrs.group}
            """
        )

    def _gen_hardswish(self, call):
        func = call.op
        data, _, _ = peel_hardswish(func.body)

        self._gen_basic_layer_items("Activation", data, call)
        self._ir_text += textwrap.dedent(
            f"""
            method={"HARDSWISH"}
            """
        )

    def _gen_qnn_hardswish(self, call):
        func = call.op
        quantize = func.body
        dequantize, _, _ = peel_hardswish(quantize.args[0])

        self._gen_basic_layer_items("Activation", dequantize.args[0], call)
        self._ir_text += textwrap.dedent(
            f"""
            method={"HARDSWISH"}
            """
        )

    def _gen_argminmax(self, argminmax, method):
        attrs = argminmax.attrs

        self._gen_basic_layer_items("ArgMinMax", argminmax.args, argminmax)
        self._ir_text += textwrap.dedent(
            f"""
            axis={attrs.axis[0]}
            select_last_index={bool(attrs.select_last_index)}
            method={method}
            """
        )

    def _gen_softplus(self, softplus):
        body = softplus.op.body
        param = body.args[0].args[0].args
        self._gen_basic_layer_items("Activation", param, softplus)
        self._ir_text += textwrap.dedent(
            f"""
            method={"SOFTPLUS"}
            """
        )

    def _gen_gelu(self, gelu):
        body = gelu.op.body
        param = body.args[0].args[0]
        self._gen_basic_layer_items("Activation", param, gelu)
        self._ir_text += textwrap.dedent(
            f"""
            method={"GELU"}
            approximate={"TANH"}
            """
        )

    def _gen_reverse_sequence(self, reverse_sequence):
        attrs = reverse_sequence.attrs

        self._gen_basic_layer_items("ReverseSequence", reverse_sequence.args, reverse_sequence)
        self._ir_text += textwrap.dedent(
            f"""
            batch_axis={int(attrs.batch_axis)}
            time_axis={int(attrs.seq_axis)}
            """
        )

    def _gen_batch_matmul(self, batch_matmul):
        attrs = batch_matmul.attrs

        self._gen_basic_layer_items("MatMul", batch_matmul.args, batch_matmul)
        self._ir_text += textwrap.dedent(
            f"""
            trans_a={bool(attrs.transpose_a)}
            trans_b={bool(attrs.transpose_b)}
            """
        )

    def _gen_negative(self, negative):
        self._gen_basic_layer_items("Negative", negative.args, negative)
        self._ir_text += "\n"

    def _gen_space_to_depth(self, space_to_depth):
        attrs = space_to_depth.attrs

        self._gen_basic_layer_items("SpaceToDepth", space_to_depth.args, space_to_depth)
        self._ir_text += textwrap.dedent(
            f"""
            block_size_x={attrs.block_size}
            block_size_y={attrs.block_size}
            """
        )

    def _gen_qnn_requant_s2d(self, call):
        func = call.op
        space_to_depth = func.body
        requant = space_to_depth.args[0]

        attrs = space_to_depth.attrs

        self._gen_basic_layer_items("SpaceToDepth", requant.args[0], call)
        self._ir_text += textwrap.dedent(
            f"""
            block_size_x={attrs.block_size}
            block_size_y={attrs.block_size}
            """
        )

    def _gen_depth_to_space(self, depth_to_space):
        attrs = depth_to_space.attrs

        self._gen_basic_layer_items("DepthToSpace", depth_to_space.args, depth_to_space)
        self._ir_text += textwrap.dedent(
            f"""
            block_size_x={attrs.block_size}
            block_size_y={attrs.block_size}
            mode={attrs.mode.upper()}
            """
        )

    def _gen_batch_to_space_nd(self, batch_to_space_nd):
        attrs = batch_to_space_nd.attrs
        block_size_x, block_size_y = attrs.block_shape
        crop_left, crop_right = attrs.crops[1]
        crop_top, crop_bottom = attrs.crops[0]

        self._gen_basic_layer_items("BatchToSpace", batch_to_space_nd.args, batch_to_space_nd)
        self._ir_text += textwrap.dedent(
            f"""
            block_size_x={block_size_x}
            block_size_y={block_size_y}
            crop_left={crop_left}
            crop_right={crop_right}
            crop_top={crop_top}
            crop_bottom={crop_bottom}
            """
        )

    def _gen_space_to_batch_nd(self, space_to_batch_nd):
        attrs = space_to_batch_nd.attrs
        block_shape = list(attrs.block_shape)
        if len(block_shape) == 1:
            block_shape += [1]
        block_size_y, block_size_x = block_shape
        pad_top, pad_bottom = attrs.paddings[0]
        if len(attrs.paddings) == 1:
            pad_left, pad_right = 0, 0
        else:
            pad_left, pad_right = attrs.paddings[1]

        self._gen_basic_layer_items("SpaceToBatch", space_to_batch_nd.args, space_to_batch_nd)
        self._ir_text += textwrap.dedent(
            f"""
            block_size_x={block_size_x}
            block_size_y={block_size_y}
            pad_left={pad_left}
            pad_right={pad_right}
            pad_top={pad_top}
            pad_bottom={pad_bottom}
            """
        )

    def _gen_gruv3(self, gruv3):
        attrs = gruv3.attrs
        _, _, weight, bias = gruv3.args

        weight_offset, weight_nbytes = self._get_offset_nbytes(weight)
        bias_offset, bias_nbytes = self._get_offset_nbytes(bias)
        weight_ttype = weight.checked_type
        bias_ttype = bias.checked_type

        _, time_steps, input_size = gruv3.args[0].checked_type.shape
        cell_size = bias.checked_type.shape[0] // 3
        self._gen_basic_layer_items("GRUv3", gruv3.args[:2], gruv3)
        self._ir_text += textwrap.dedent(
            f"""
            weights_type={weight_ttype.dtype}
            weights_offset={weight_offset}
            weights_size={weight_nbytes}
            weights_shape={_verify_shape(weight_ttype.shape)}
            biases_type={bias_ttype.dtype}
            biases_offset={bias_offset}
            biases_size={bias_nbytes}
            biases_shape={[int(x) for x in bias_ttype.shape if x != 1]}
            activations=[{attrs.activations}]
            cell_size={int(cell_size)}
            direction=forward
            input_size={int(input_size)}
            out_sequence=[{attrs.out_sequence}]
            time_steps={int(time_steps)}
            """
        )

    def _gen_take(self, take):
        attrs = take.attrs
        batch_dims = attrs.batch_dims
        axis = attrs.axis

        self._gen_basic_layer_items("Gather", take.args[:2], take)
        self._ir_text += textwrap.dedent(
            f"""
            axis={int(axis)}
            batch_dims={int(batch_dims)}
            """
        )

    def _gen_logical(self, logical, method):
        self._gen_basic_layer_items("Logical", logical.args, logical)
        self._ir_text += textwrap.dedent(
            f"""
            method={method}
            """
        )

    def _gen_bitwise(self, bitwise, method):
        self._gen_basic_layer_items("Bitwise", bitwise.args, bitwise)
        self._ir_text += textwrap.dedent(
            f"""
            method={method}
            """
        )

    def _gen_where(self, where):
        self._gen_basic_layer_items("Where", where.args, where)
        self._ir_text += textwrap.dedent(
            """
            """
        )

    def _gen_layernorm0(self, layernorm):
        add = layernorm.op.body
        mul, bias = unpack_commutative_args(add)
        div, weight = unpack_commutative_args(mul)
        epsilon_add = div.args[1].args[0]
        _, epsilon = unpack_commutative_args(epsilon_add)
        epsilon = float(epsilon.data.numpy())
        axis = div.args[0].args[1].attrs.axis

        weight_offset, weight_nbytes = self._get_offset_nbytes(weight)
        bias_offset, bias_nbytes = self._get_offset_nbytes(bias)
        weight_ttype = weight.checked_type
        bias_ttype = bias.checked_type

        self._gen_basic_layer_items("LayerNorm", layernorm.op.params, layernorm)
        self._ir_text += textwrap.dedent(
            f"""
            weights_type={weight_ttype.dtype}
            weights_offset={weight_offset}
            weights_size={weight_nbytes}
            weights_shape={_verify_shape(weight_ttype.shape)}
            biases_type={bias_ttype.dtype}
            biases_offset={bias_offset}
            biases_size={bias_nbytes}
            biases_shape={_verify_shape(bias_ttype.shape)}
            axis={[int(x) for x in axis]}
            epsilon={epsilon}
            """
        )

    def _gen_layernorm1(self, layernorm):
        add = layernorm.op.body
        mul, bias = unpack_commutative_args(add)
        rsqrt_mul, weight = unpack_commutative_args(mul)
        inp_mean, rsqrt = unpack_commutative_args(rsqrt_mul, "rsqrt")
        _, epsilon = unpack_commutative_args(rsqrt.args[0])
        epsilon = float(epsilon.data.numpy())
        axis = inp_mean.args[1].attrs.axis

        weight_offset, weight_nbytes = self._get_offset_nbytes(weight)
        bias_offset, bias_nbytes = self._get_offset_nbytes(bias)
        weight_ttype = weight.checked_type
        bias_ttype = bias.checked_type

        self._gen_basic_layer_items("LayerNorm", layernorm.op.params, layernorm)
        self._ir_text += textwrap.dedent(
            f"""
            weights_type={weight_ttype.dtype}
            weights_offset={weight_offset}
            weights_size={weight_nbytes}
            weights_shape={_verify_shape(weight_ttype.shape)}
            biases_type={bias_ttype.dtype}
            biases_offset={bias_offset}
            biases_size={bias_nbytes}
            biases_shape={_verify_shape(bias_ttype.shape)}
            axis={[int(x) for x in axis]}
            epsilon={epsilon}
            """
        )

    def _gen_ctc_greedy_decoder(self, ctc):
        attrs = ctc.attrs

        self._gen_basic_layer_items("CTCGreedyDecoder", ctc.args, ctc)
        self._ir_text += textwrap.dedent(
            f"""
            merge_repeated={bool(attrs.merge_repeated)}
            """
        )

    def _gen_fake_quant_min_max_vars(self, fake_quant):
        attrs = fake_quant.attrs

        self._gen_basic_layer_items("FakeQuantWithMinMaxVars", fake_quant.args, fake_quant)
        self._ir_text += textwrap.dedent(
            f"""
            max={attrs.maximum}
            min={attrs.minimum}
            narrow_range={bool(attrs.narrow_range)}
            num_bits={attrs.num_bits}
            """
        )

    def _gen_mean_variance_norm(self, norm):
        if norm.op.body.op == relay.op.get("multiply"):
            _, rsqrt = unpack_commutative_args(norm.op.body, "rsqrt")
            epsilon_add = rsqrt.args[0]
        else:
            epsilon_add = norm.op.body.args[1].args[0]

        add_node, epsilon = unpack_commutative_args(epsilon_add)
        epsilon = epsilon.data.numpy()
        axis = add_node.attrs.axis
        axis = [int(x) for x in axis]

        # currently MeanVarianceNormalization is not supported by gbuilder
        # use layernorm instead
        inp_shape = norm.args[0].checked_type.shape
        dtype = norm.args[0].checked_type.dtype
        channel = int(inp_shape[-1])
        weight = np.ones([channel], dtype=dtype)
        bias = np.zeros([channel], dtype=dtype)
        weight = relay.const(nd.array(weight))
        bias = relay.const(nd.array(bias))

        weight_offset, weight_nbytes = self._get_offset_nbytes(weight)
        bias_offset, bias_nbytes = self._get_offset_nbytes(bias)

        self._gen_basic_layer_items("LayerNorm", norm.op.params, norm)
        self._ir_text += textwrap.dedent(
            f"""
            weights_type={dtype}
            weights_offset={weight_offset}
            weights_size={weight_nbytes}
            weights_shape={[channel]}
            biases_type={dtype}
            biases_offset={bias_offset}
            biases_size={bias_nbytes}
            biases_shape={[channel]}
            axis={[int(x) for x in axis]}
            epsilon={epsilon}
            """
        )

    def _gen_round(self, call):
        self._gen_basic_layer_items("Round", call.args, call)
        self._ir_text += textwrap.dedent(
            """
            """
        )

    def _gen_topk(self, topk):
        attrs = topk.attrs
        inp_shape = topk.args[0].checked_type.shape
        axis = attrs.axis
        axis = axis + len(inp_shape) if axis == -1 else axis
        k = attrs.k
        k = inp_shape[axis] if k < 1 else k
        largest = not attrs.is_ascend

        self._gen_basic_layer_items("TopK", topk.args, topk)

        self._ir_text += textwrap.dedent(
            f"""
            k={k}
            axis={axis}
            largest={largest}
            sorted=true
            select_index=last
            """
        )

    def _gen_simple_op(self, call, layer_type):
        self._gen_basic_layer_items(layer_type, call.args, call)
        self._ir_text += textwrap.dedent(
            """
            """
        )

    def _gen_simple_qnn_op(self, call, layer_type):
        self._gen_basic_layer_items(layer_type, call.args[0], call)
        self._ir_text += textwrap.dedent(
            """
            """
        )

    def _gen_crop_and_resize(self, call):
        attrs = call.attrs
        method = "BILINEAR"
        if attrs.method == "nearest_neighbor":
            method = "NEAREST"

        self._gen_basic_layer_items("CropAndResize", call.args, call)
        self._ir_text += textwrap.dedent(
            f"""
            crop_size={attrs.crop_size}
            method={method}
            extrapolation_value={attrs.extrapolation_value}
            """
        )

    def _gen_mirror_pad(self, call):
        attrs = call.attrs
        pad_mode = attrs.mode.upper()
        self._gen_basic_layer_items("Pad", call.args, call)
        self._ir_text += textwrap.dedent(
            f"""
            constant_value=0
            mode={pad_mode}
            pads={[[int(y) for y in x] for x in attrs.pad_width]}
            """
        )

    def _gen_qnn_mirror_pad(self, call):
        func = call.op
        requantize = func.body
        mirror_pad = requantize.args[0]

        attrs = mirror_pad.attrs
        pad_mode = attrs.mode.upper()
        self._gen_basic_layer_items("Pad", mirror_pad.args[0], call)
        self._ir_text += textwrap.dedent(
            f"""
            constant_value=0
            mode={pad_mode}
            pads={[[int(y) for y in x] for x in attrs.pad_width]}
            """
        )

    def _gen_scatter_elements(self, call):
        attrs = call.attrs
        reduction = attrs.reduction.upper()
        if reduction == "UPDATE":
            reduction = "NONE"

        self._gen_basic_layer_items("ScatterElements", call.args, call)
        self._ir_text += textwrap.dedent(
            f"""
            axis={attrs.axis}
            reduction={reduction}
            """
        )

    def _gen_scatter_nd(self, call):
        reduce_method = str(call.attrs.mode).upper()
        if reduce_method == "UPDATE":
            reduce_method = "NONE"
        indices = call.args[1].data.numpy()
        idim = len(indices.shape)
        order = list(range(1, idim, 1)) + [0]
        indices = np.transpose(indices, order)
        new_indices = relay.Constant(nd.array(indices))
        relay.transform.InferTypeLocal(new_indices)
        self._gen_constant(new_indices)
        data = call.args[0]
        value = call.args[2]
        if isinstance(data, relay.Constant):
            self._gen_constant(data)
        if isinstance(value, relay.Constant):
            self._gen_constant(value)

        input_names = []
        input_shapes = []
        input_types = []
        inputs = list(call.args)
        inputs[1] = new_indices
        for arg in inputs:
            input_names.append(self._get_or_alloc_var_name(arg))
            input_shapes.append(arg.checked_type.shape)
            input_types.append(arg.checked_type.dtype)

        output_names = []
        output_shapes = []
        output_types = []
        output_names.append(self._get_or_alloc_var_name(call))
        output_shapes.append(call.checked_type.shape)
        output_types.append(call.checked_type.dtype)

        layer_idx = self._get_layer_idx()
        input_types = [inp_type if inp_type != "bool" else "uint8" for inp_type in input_types]
        output_types = [out_type if out_type != "bool" else "uint8" for out_type in output_types]

        self._ir_text += textwrap.dedent(
            f"""
            layer_id={layer_idx}
            layer_name={layer_idx}_scatternd
            layer_type=ScatterND
            layer_bottom=[{", ".join(input_names)}]
            layer_bottom_shape={_verify_shape(input_shapes)}
            layer_bottom_type=[{", ".join(input_types)}]
            layer_top=[{", ".join(output_names)}]
            layer_top_shape={_verify_shape(output_shapes)}
            layer_top_type=[{", ".join(output_types)}]
            reduction={reduce_method}
            """
        )

    def _gen_gather(self, call):
        attrs = call.attrs

        self._gen_basic_layer_items("GatherElements", call.args, call)
        self._ir_text += textwrap.dedent(
            f"""
            axis={attrs.axis}
            """
        )

    def _gen_gather_nd(self, call):
        indices = call.args[1].data.numpy()
        idim = len(indices.shape)
        order = list(range(1, idim, 1)) + [0]
        indices = np.transpose(indices, order)
        new_indices = relay.Constant(nd.array(indices))
        relay.transform.InferTypeLocal(new_indices)
        self._gen_constant(new_indices)
        data = call.args[0]
        if isinstance(data, relay.Constant):
            self._gen_constant(data)

        input_names = []
        input_shapes = []
        input_types = []
        inputs = list(call.args)
        inputs[1] = new_indices
        for arg in inputs:
            input_names.append(self._get_or_alloc_var_name(arg))
            input_shapes.append(arg.checked_type.shape)
            input_types.append(arg.checked_type.dtype)

        output_names = []
        output_shapes = []
        output_types = []
        output_names.append(self._get_or_alloc_var_name(call))
        output_shapes.append(call.checked_type.shape)
        output_types.append(call.checked_type.dtype)

        layer_idx = self._get_layer_idx()
        input_types = [inp_type if inp_type != "bool" else "uint8" for inp_type in input_types]
        output_types = [out_type if out_type != "bool" else "uint8" for out_type in output_types]

        self._ir_text += textwrap.dedent(
            f"""
            layer_id={layer_idx}
            layer_name={layer_idx}_gathernd
            layer_type=GatherND
            layer_bottom=[{", ".join(input_names)}]
            layer_bottom_shape={_verify_shape(input_shapes)}
            layer_bottom_type=[{", ".join(input_types)}]
            layer_top=[{", ".join(output_names)}]
            layer_top_shape={_verify_shape(output_shapes)}
            layer_top_type=[{", ".join(output_types)}]
            batch_dims={call.attrs.batch_dims}
            """
        )

    def _gen_non_max_suppression(self, call):
        data, valid_count, _, max_output_size, iou_threshold = call.args

        iou_threshold = float(iou_threshold.data.numpy())
        max_output_size = int(max_output_size.data.numpy())

        # valid_count [batch]
        # data [batch, box_num, 5]

        split = relay.split(data, [1], axis=2)
        split_tuple = split.tuple_value
        score = relay.TupleGetItem(split_tuple, 0)
        anchor_boxes = relay.TupleGetItem(split_tuple, 1)
        relay.transform.InferTypeLocal(split_tuple)
        relay.transform.InferTypeLocal(score)
        relay.transform.InferTypeLocal(anchor_boxes)
        self._tuple2tgn[split_tuple] = [score, anchor_boxes]

        batch, box_num, _ = data.checked_type.shape
        batch = int(batch)
        box_num = int(box_num)
        self._gen_split(split_tuple)
        score = relay.reshape(score, (batch, box_num))
        relay.transform.InferTypeLocal(score)
        self._gen_reshape(score)
        # compass NMS have 4 inputs
        # boxes [batch, box_num, 4]
        # boxnum_perclass [batch, class_num]
        # total_class_num [batch, 1]
        # scores [batch, box_num]
        # 4 outputs
        # batch_nms_boxes
        # batch_nms_boxNum_perClass
        # batch_nms_scores
        # batch_keep

        # assume only 1 class
        total_class_num = relay.const(
            nd.array(np.array([1] * batch, dtype=np.int32).reshape([batch, 1]))
        )
        boxnum_perclass = relay.reshape(valid_count, (batch, 1))
        relay.transform.InferTypeLocal(total_class_num)
        relay.transform.InferTypeLocal(boxnum_perclass)
        self._gen_reshape(boxnum_perclass)
        self._gen_constant(total_class_num)

        output_names = []
        output_shapes = []
        output_types = []
        output_names.append(self._get_or_alloc_var_name(call))
        output_shapes.append(call.checked_type.shape)
        output_types.append(call.checked_type.dtype)

        layer_idx = self._get_layer_idx()

        inputs = [anchor_boxes, boxnum_perclass, total_class_num, score]
        input_names = []
        input_shapes = []
        input_types = []
        for arg in inputs:
            input_names.append(self._get_or_alloc_var_name(arg))
            input_shapes.append(arg.checked_type.shape)
            input_types.append(arg.checked_type.dtype)

        output_names = [
            f"{layer_idx}_nms_box",
            f"{layer_idx}_nms_boxNum_perClass",
            f"{layer_idx}_nms_scores",
            f"{layer_idx}_nms_keep",
        ]

        if max_output_size <= 0:
            max_output_size = box_num
        else:
            max_output_size = min(max_output_size, box_num)
        output_types = [anchor_boxes.checked_type.dtype, "int32", score.checked_type.dtype, "int32"]
        output_shapes = [
            [batch, max_output_size, 4],
            [batch, 1],
            [batch, max_output_size],
            [batch, 1],
        ]

        self._ir_text += textwrap.dedent(
            f"""
            layer_id={layer_idx}
            layer_name={layer_idx}_nms
            layer_type=NMS
            layer_bottom=[{", ".join(input_names)}]
            layer_bottom_shape={_verify_shape(input_shapes)}
            layer_bottom_type=[{", ".join(input_types)}]
            layer_top=[{", ".join(output_names)}]
            layer_top_shape={_verify_shape(output_shapes)}
            layer_top_type=[{", ".join(output_types)}]
            max_output_size={max_output_size}
            iou_threshold={iou_threshold}
            """
        )
        reshape_id = self._get_layer_idx()
        reshape_name = f"{reshape_id}_reshape_score"
        self._ir_text += textwrap.dedent(
            f"""
            layer_id={reshape_id}
            layer_name={reshape_id}_reshape
            layer_type=Reshape
            layer_bottom=[{f"{layer_idx}_nms_scores"}]
            layer_bottom_shape=[{[batch, max_output_size]}]
            layer_bottom_type=[{anchor_boxes.checked_type.dtype}]
            layer_top=[{reshape_name}]
            layer_top_shape=[{[batch, max_output_size, 1]}]
            layer_top_type=[{anchor_boxes.checked_type.dtype}]
            shape={[batch, max_output_size, 1]}
            """
        )

        nms_name = self._get_or_alloc_var_name(call)

        concat_id = self._get_layer_idx()
        self._ir_text += textwrap.dedent(
            f"""
            layer_id={concat_id}
            layer_name={concat_id}_concat
            layer_type=Concat
            layer_bottom=[{reshape_name},{f"{layer_idx}_nms_box"}]
            layer_bottom_shape=[{[batch, max_output_size, 1]},{[batch, max_output_size, 4]}]
            layer_bottom_type=[{anchor_boxes.checked_type.dtype},{anchor_boxes.checked_type.dtype}]
            layer_top=[{nms_name}]
            layer_top_shape=[{[batch, max_output_size, 5]}]
            layer_top_type=[{anchor_boxes.checked_type.dtype}]
            axis=2
            """
        )

    def _gen_get_valid_counts(self, call):
        _, thres = call.args
        thres = float(thres.data.numpy())
        attrs = call.attrs

        id_index = attrs.id_index
        score_index = attrs.score_index

        self._gen_basic_layer_items("GetValidCount", call.args[:1], call)
        self._ir_text += textwrap.dedent(
            f"""
            score_threshold={thres}
            id_index={id_index}
            score_index={score_index}
            """
        )

    def _gen_multibox_transform_loc(self, call):
        attrs = call.attrs
        score_threshold = attrs.threshold
        variances = [float(val) for val in attrs.variances]

        self._gen_basic_layer_items("MultiboxTransformLoc", call.args, call)
        self._ir_text += textwrap.dedent(
            f"""
            score_threshold={score_threshold}
            variance={variances}
            """
        )

    def _gen_roi_align(self, call):
        attrs = call.attrs

        data, rois = call.args
        data_type = rois.checked_type.dtype
        data_shape = data.checked_type.shape
        rois_shape = rois.checked_type.shape
        rois_type = rois.checked_type.dtype
        rois_shape = [int(val) for val in rois_shape]
        rois_num = rois_shape[0]

        splits = relay.split(rois, indices_or_sections=5, axis=-1).tuple_value
        val_idx = [0, 2, 1, 4, 3]
        val_expr = [relay.TupleGetItem(splits, idx) for idx in val_idx]
        concat = relay.concatenate(val_expr, -1)

        split_id = self._get_layer_idx()
        concat_id = self._get_layer_idx()
        roi_align_id = self._get_layer_idx()
        rois_name = self._get_or_alloc_var_name(rois)
        split_name = [rois_name + "_split_" + f"{split_id}_" + str(idx) for idx in range(5)]
        new_idx = [0, 2, 1, 4, 3]
        concat_input_name = [split_name[idx] for idx in new_idx]
        str_shape = [f"[{rois_num}, 1]"] * 5
        split_shape = ",".join(str_shape)
        concat_out_name = self._get_or_alloc_var_name(concat)

        mode = attrs.mode
        pooled_size = attrs.pooled_size
        pooled_size = [int(val) for val in pooled_size]
        sample_ratio = attrs.sample_ratio
        spatial_scale = attrs.spatial_scale

        self._ir_text += textwrap.dedent(
            f"""
        layer_id={split_id}
        layer_name={split_id}_split
        layer_type=Split
        layer_bottom=[{rois_name}]
        layer_bottom_shape=[{_verify_shape(rois_shape)}]
        layer_bottom_type=[{rois_type}]
        layer_top=[{",".join(split_name)}]
        layer_top_shape=[{split_shape}]
        layer_top_type=[{",".join([rois_type] * 5)}]
        splits=[1, 1, 1, 1, 1]
        axis=-1
        """
        )

        self._ir_text += textwrap.dedent(
            f"""
        layer_id={concat_id}
        layer_name={concat_id}_concat
        layer_type=Concat
        layer_bottom=[{",".join(concat_input_name)}]
        layer_bottom_shape=[{split_shape}]
        layer_bottom_type=[{",".join([rois_type] * 5)}]
        layer_top=[{concat_out_name}]
        layer_top_shape=[{[rois_num, 5]}]
        layer_top_type=[{rois_type}]
        axis=-1
        """
        )

        self._ir_text += textwrap.dedent(
            f"""
        layer_id={roi_align_id}
        layer_name={roi_align_id}_roialign
        layer_type=RoiAlign
        layer_bottom=[{self._get_or_alloc_var_name(data)},{concat_out_name}]
        layer_bottom_shape=[{_verify_shape(data_shape)},{[rois_num, 5]}]
        layer_bottom_type=[{data_type},{rois_type}]
        layer_top=[{self._get_or_alloc_var_name(call)}]
        layer_top_shape=[{_verify_shape(call.checked_type.shape)}]
        layer_top_type=[{call.checked_type.dtype}]
        method={str(mode).upper()}
        pooled_shape={pooled_size}
        spatial_scale_value=[{spatial_scale}, {spatial_scale}]
        sample=[{sample_ratio}, {sample_ratio}]
        coordinate_transformation_mode=OUTPUT_HALF_PIXEL
        """
        )

    def _gen_roi_pool(self, call):
        attrs = call.attrs

        data, rois = call.args
        data_type = rois.checked_type.dtype
        data_shape = data.checked_type.shape
        rois_shape = rois.checked_type.shape
        rois_type = rois.checked_type.dtype
        rois_shape = [int(val) for val in rois_shape]
        rois_num = rois_shape[0]

        splits = relay.split(rois, indices_or_sections=5, axis=-1).tuple_value
        val_idx = [0, 2, 1, 4, 3]
        val_expr = [relay.TupleGetItem(splits, idx) for idx in val_idx]
        concat = relay.concatenate(val_expr, -1)

        split_id = self._get_layer_idx()
        concat_id = self._get_layer_idx()
        roi_align_id = self._get_layer_idx()
        rois_name = self._get_or_alloc_var_name(rois)
        split_name = [rois_name + "_split_" + f"{split_id}_" + str(idx) for idx in range(5)]
        new_idx = [0, 2, 1, 4, 3]
        concat_input_name = [split_name[idx] for idx in new_idx]
        str_shape = [f"[{rois_num}, 1]"] * 5
        split_shape = ",".join(str_shape)
        concat_out_name = self._get_or_alloc_var_name(concat)

        pooled_size = attrs.pooled_size
        pooled_size = [int(val) for val in pooled_size]
        spatial_scale = attrs.spatial_scale

        self._ir_text += textwrap.dedent(
            f"""
        layer_id={split_id}
        layer_name={split_id}_split
        layer_type=Split
        layer_bottom=[{rois_name}]
        layer_bottom_shape=[{_verify_shape(rois_shape)}]
        layer_bottom_type=[{rois_type}]
        layer_top=[{",".join(split_name)}]
        layer_top_shape=[{split_shape}]
        layer_top_type=[{",".join([rois_type] * 5)}]
        splits=[1, 1, 1, 1, 1]
        axis=-1
        """
        )

        self._ir_text += textwrap.dedent(
            f"""
        layer_id={concat_id}
        layer_name={concat_id}_concat
        layer_type=Concat
        layer_bottom=[{",".join(concat_input_name)}]
        layer_bottom_shape=[{split_shape}]
        layer_bottom_type=[{",".join([rois_type] * 5)}]
        layer_top=[{concat_out_name}]
        layer_top_shape=[{[rois_num, 5]}]
        layer_top_type=[{rois_type}]
        axis=-1
        """
        )

        self._ir_text += textwrap.dedent(
            f"""
        layer_id={roi_align_id}
        layer_name={roi_align_id}_maxroipool
        layer_type=MaxRoiPool
        layer_bottom=[{self._get_or_alloc_var_name(data)},{concat_out_name}]
        layer_bottom_shape=[{_verify_shape(data_shape)},{[rois_num, 5]}]
        layer_bottom_type=[{data_type},{rois_type}]
        layer_top=[{self._get_or_alloc_var_name(call)}]
        layer_top_shape=[{_verify_shape(call.checked_type.shape)}]
        layer_top_type=[{call.checked_type.dtype}]
        pooled_shape={pooled_size}
        spatial=[{spatial_scale}, {spatial_scale}]
        """
        )

    def _gen_custom_op(self, call, func):
        """
        custom op codegen func need to return:
            layer_type: the layer type name.
            inp: the inputs of this op.
            constants: {name: constant} dict to get constant offset.
            attr_text: all attr textwrap of this op.
        """
        layer_type, inp, constants, attr_text = func(call)

        self._gen_basic_layer_items(layer_type, inp, call)
        for k, v in constants.items():
            offset, nbytes = self._get_offset_nbytes(v)
            self._ir_text += textwrap.dedent(
                f"""
                {k}_type={v.data.dtype}
                {k}_shape={list(v.data.shape)}
                {k}_offset={offset}
                {k}_size={nbytes}"""
            )
        self._ir_text += attr_text

    def _gen_left_shift(self, left_shift):
        self._gen_basic_layer_items("BitShift", left_shift.args, left_shift)
        self._ir_text += textwrap.dedent(
            """
            direction=LEFT
            """
        )

    def _gen_right_shift(self, right_shift):
        self._gen_basic_layer_items("BitShift", right_shift.args, right_shift)
        self._ir_text += textwrap.dedent(
            """
            direction=RIGHT
            """
        )

    def _gen_erf(self, erf):
        self._gen_basic_layer_items("Erf", erf.args, erf)
        self._ir_text += textwrap.dedent(
            """
            """
        )

    def _gen_trunc(self, trunc):
        self._gen_basic_layer_items("Trunc", trunc.args, trunc)
        self._ir_text += textwrap.dedent(
            """
            """
        )

    def _gen_cumsum(self, cumsum):
        attrs = cumsum.attrs
        axis = attrs.axis
        exclusive = "true" if attrs.exclusive else "false"
        reverse = "false"

        self._gen_basic_layer_items("Cumulate", cumsum.args, cumsum)
        self._ir_text += textwrap.dedent(
            f"""
            method=SUM
            axis={axis}
            exclusive={exclusive}
            reverse={reverse}
            """
        )

    def _gen_cumprod(self, cumprod):
        attrs = cumprod.attrs
        axis = attrs.axis
        exclusive = "true" if attrs.exclusive else "false"
        reverse = "false"

        self._gen_basic_layer_items("Cumulate", cumprod.args, cumprod)
        self._ir_text += textwrap.dedent(
            f"""
            method=PROD
            axis={axis}
            exclusive={exclusive}
            reverse={reverse}
            """
        )

    def _gen_meshgrid(self, meshgrid):
        attrs = meshgrid.attrs

        self._gen_basic_layer_items("Meshgrid", meshgrid.args, meshgrid)
        self._ir_text += textwrap.dedent(
            f"""
            indexing={attrs.indexing}
            sparse=false
            copy=true
            """
        )

    def _gen_detection_output(self, call):
        attrs = call.attrs

        self._gen_basic_layer_items("DetectionOutput", call.args, call)
        self._ir_text += textwrap.dedent(
            f"""
            image_width={int(attrs.image_width)}
            image_height={int(attrs.image_height)}
            score_threshold={float(attrs.score_threshold)}
            variance={[float(v) for v in attrs.variance]}
            """
        )

    def _gen_nms(self, call):
        attrs = call.attrs

        self._gen_basic_layer_items("NMS", call.args, call)
        self._ir_text += textwrap.dedent(
            f"""
            image_width={int(attrs.image_width)}
            image_height={int(attrs.image_height)}
            max_output_size={int(attrs.max_output_size)}
            center_point_box={int(attrs.center_point_box)}
            method={attrs.method}
            iou_threshold={float(attrs.iou_threshold)}
            soft_nms_sigma={float(attrs.soft_nms_sigma)}
            score_threshold={float(attrs.score_threshold)}
            """
        )

    def _gen_decode_box(self, call):
        attrs = call.attrs
        ycenter, xcenter, ha_data, wa_data = call.args[2:]
        # ycenter
        ycenter_dtype = ycenter.checked_type.dtype
        ycenter_shape = ycenter.checked_type.shape
        ycenter_offset, ycenter_nbytes = self._get_offset_nbytes(ycenter)
        # xcenter
        xcenter_dtype = xcenter.checked_type.dtype
        xcenter_shape = xcenter.checked_type.shape
        xcenter_offset, xcenter_nbytes = self._get_offset_nbytes(xcenter)
        # ha_data
        ha_dtype = ha_data.checked_type.dtype
        ha_shape = ha_data.checked_type.shape
        ha_offset, ha_nbytes = self._get_offset_nbytes(ha_data)
        # wa_data
        wa_dtype = wa_data.checked_type.dtype
        wa_shape = wa_data.checked_type.shape
        wa_offset, wa_nbytes = self._get_offset_nbytes(wa_data)

        self._gen_basic_layer_items("DecodeBox", call.args[:2], call)
        self._ir_text += textwrap.dedent(
            f"""
            ycenter_type={ycenter_dtype}
            ycenter_offset={ycenter_offset}
            ycenter_size={ycenter_nbytes}
            ycenter_shape={_verify_shape(ycenter_shape)}
            xcenter_type={xcenter_dtype}
            xcenter_offset={xcenter_offset}
            xcenter_size={xcenter_nbytes}
            xcenter_shape={_verify_shape(xcenter_shape)}
            ha_type={ha_dtype}
            ha_offset={ha_offset}
            ha_size={ha_nbytes}
            ha_shape={_verify_shape(ha_shape)}
            wa_type={wa_dtype}
            wa_offset={wa_offset}
            wa_size={wa_nbytes}
            wa_shape={_verify_shape(wa_shape)}
            image_width={int(attrs.image_width)}
            image_height={int(attrs.image_height)}
            max_box_num={int(attrs.max_box_num)}
            class_num={int(attrs.class_num)}
            score_threshold={float(attrs.score_threshold)}
            """
        )

    def _gen_dequantize(self, call):
        scale = call.args[1]
        scale_dtype = scale.checked_type.dtype
        scale_shape = scale.checked_type.shape
        scale_offset, scale_nbytes = self._get_offset_nbytes(scale)

        zero_point = call.args[2]
        zp_dtype = zero_point.checked_type.dtype
        zp_shape = zero_point.checked_type.shape
        zp_offset, zp_nbytes = self._get_offset_nbytes(zero_point)

        self._gen_basic_layer_items("DeQuantize", call.args[0], call)
        self._ir_text += textwrap.dedent(
            f"""
            quantize_scale_type={scale_dtype}
            quantize_scale_offset={scale_offset}
            quantize_scale_size={scale_nbytes}
            quantize_scale_shape={scale_shape}
            quantize_zp_type={zp_dtype}
            quantize_zp_offset={zp_offset}
            quantize_zp_size={zp_nbytes}
            quantize_zp_shape={zp_shape}
            """
        )

    def _gen_quantize(self, call):
        scale = call.args[1]
        scale_dtype = scale.checked_type.dtype
        scale_shape = scale.checked_type.shape
        scale_offset, scale_nbytes = self._get_offset_nbytes(scale)

        zero_point = call.args[2]
        zp_dtype = zero_point.checked_type.dtype
        zp_shape = zero_point.checked_type.shape
        zp_offset, zp_nbytes = self._get_offset_nbytes(zero_point)

        self._gen_basic_layer_items("Quantize", call.args[0], call)
        self._ir_text += textwrap.dedent(
            f"""
            quantize_scale_type={scale_dtype}
            quantize_scale_offset={scale_offset}
            quantize_scale_size={scale_nbytes}
            quantize_scale_shape={scale_shape}
            quantize_zp_type={zp_dtype}
            quantize_zp_offset={zp_offset}
            quantize_zp_size={zp_nbytes}
            quantize_zp_shape={zp_shape}
            """
        )

    def _gen_dense_add_activation(self, call):
        func = call.op
        activation = func.body
        if activation.op.name == "clip":
            with_activation = "CLIP"
        elif activation.op.name == "nn.relu":
            with_activation = "RELU"
        elif activation.op.name == "nn.leaky_relu":
            with_activation = "LEAKYRELU"

        add = activation.args[0]
        dense, bias = unpack_commutative_args(add)
        weight = dense.args[1]

        weight_offset, weight_nbytes = self._get_offset_nbytes(weight)
        bias_offset, bias_nbytes = self._get_offset_nbytes(bias)
        weight_ttype = weight.checked_type
        bias_ttype = bias.checked_type
        output_shape = _verify_shape(call.checked_type.shape)

        self._gen_basic_layer_items("FullyConnected", dense.args[0], call)
        self._ir_text += textwrap.dedent(
            f"""
            weights_type={weight_ttype.dtype}
            weights_offset={weight_offset}
            weights_size={weight_nbytes}
            weights_shape={_verify_shape(weight_ttype.shape)}
            biases_type={bias_ttype.dtype}
            biases_offset={bias_offset}
            biases_size={bias_nbytes}
            biases_shape={[int(x) for x in bias_ttype.shape if x != 1] or [1]}
            num_output={output_shape[-1]}
            with_activation={with_activation}
            """
        )
