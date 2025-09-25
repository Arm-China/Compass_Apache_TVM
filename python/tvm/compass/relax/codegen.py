# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Compass IR codegen of Relax."""
import os
import numpy as np
import tvm
from tvm import relax, ir
from tvm.relax.analysis import get_var2val
from AIPUBuilder.core import Tensor, TensorShape, Dtype, Graph, Node, OpType, register_optype
from AIPUBuilder._C._core import Quantization
from .builder import ops
from .config import CompassConfig
from .op import CODEGEN_CUSTOM_OP_DICT
from .utils import unpack_commutative_args


_DTYPE_DICT = {
    bool: Dtype.BOOL,
    "bool": Dtype.BOOL,
    "float16": Dtype.FP16,
    "float32": Dtype.FP32,
    "float64": Dtype.FP64,
    "int16": Dtype.INT16,
    "int32": Dtype.INT32,
    "int64": Dtype.INT64,
    "int8": Dtype.INT8,
    "uint8": Dtype.UINT8,
    "uint16": Dtype.UINT16,
    "uint32": Dtype.UINT32,
}


def _create_tensor(name, data, quantization=None):
    tensor = Tensor(name)
    tensor.set_numpy(data)
    if quantization:
        tensor.quantization = quantization
    return tensor


def _find_op_from_bindings(bindings, names):
    names = names if isinstance(names, (tuple, list)) else [names]
    ret = [None] * len(names)
    for bind in bindings:
        value = bind.value
        if not isinstance(value, relax.Call):
            continue
        op_name = value.op.name.split(".")[-1]
        if op_name in names:
            ret[names.index(op_name)] = value
    return ret[0] if len(ret) == 1 else ret


def _find_all_same_op_from_bindings(bindings, name):
    ret = []
    for bind in bindings:
        value = bind.value
        if not isinstance(value, relax.Call):
            continue
        op_name = value.op.name.split(".")[-1]
        if op_name == name:
            ret.append(value)
    return ret


def _get_quant(scale, zero_point):
    scale = scale.data.numpy()
    zero_point = zero_point.data.numpy()
    scale = float(scale) if len(scale.shape) == 0 else scale
    zero_point = int(zero_point) if len(zero_point.shape) == 0 else zero_point
    out = Quantization(scale, zero_point)
    out.unquantifiable = True
    return out


def _is_dq(x):
    return x.op == ir.Op.get("relax.dequantize")


def _get_quant_info(func):
    quant_info = {}
    var2val = get_var2val(func)
    bindings = func.body.blocks[0].bindings
    for bind in bindings:
        value = bind.value
        if not isinstance(value, relax.Call):
            continue
        op_name = value.op.name.split(".")[-1]
        if op_name in ["conv2d", "conv2d_transpose"]:
            inp_dq = var2val[value.args[0]]
            quant_info["input"] = _get_quant(*inp_dq.args[1:])
            weight_dq = var2val[value.args[1]]
            quant_info["weight"] = _get_quant(*weight_dq.args[1:])
        if op_name in ["add", "subtract", "multiply"]:
            if value.args[0] in var2val and _is_dq(var2val[value.args[0]]):
                lhs_dq = var2val[value.args[0]]
                quant_info["lhs"] = _get_quant(*lhs_dq.args[1:])
            if value.args[1] in var2val and _is_dq(var2val[value.args[1]]):
                rhs_dq = var2val[value.args[1]]
                quant_value = _get_quant(*rhs_dq.args[1:])
                quant_info["rhs"] = quant_value
                if op_name == "add":
                    quant_info["bias"] = quant_value
        if op_name == "matmul":
            weight_dq = var2val[value.args[1]]
            quant_info["weight"] = _get_quant(*weight_dq.args[1:])
        if op_name == "prelu":
            if value.args[1] in var2val and _is_dq(var2val[value.args[1]]):
                alpha_dq = var2val[value.args[1]]
                quant_info["alpha"] = _get_quant(*alpha_dq.args[1:])
    quantize = _find_op_from_bindings(bindings, "quantize")
    quant_info["out"] = _get_quant(*quantize.args[1:])
    return quant_info


def _broadcast_inps(origin_inps):
    if origin_inps[0].shape == origin_inps[1].shape:
        return origin_inps
    shapes = [list(x.shape) for x in origin_inps]
    max_dim = max(len(shape) for shape in shapes)
    inps = []
    new_shapes = []
    for idx, shape in enumerate(shapes):
        inp = origin_inps[idx]
        new_shape = shape
        if len(shape) != max_dim:
            new_shape = [1] * (max_dim - len(shape)) + shape
            if inp.op.type == OpType.Constant:
                data = inp.op.constants["weights"].np
                inp.op.constants["weights"].set_numpy(data.reshape(new_shape))
                inp.set_numpy(data.reshape(new_shape))  # A workaround to set inp shape.
            else:
                inp = ops.reshape(inp, new_shape)
        inps.append(inp)
        new_shapes.append(new_shape)

    new_shape = []
    for dim in range(max_dim):
        cur_dim_shape = [shape[dim] for shape in new_shapes]
        dim_max_size = max(cur_dim_shape)
        new_shape.append(dim_max_size)

    new_inps = []
    for inp, shape in zip(inps, new_shapes):
        new_inp = inp
        if shape != new_shape:
            reps = [dim0 // dim1 for dim0, dim1 in zip(new_shape, shape)]
            if inp.op.type == OpType.Constant:
                data = inp.op.constants["weights"].np
                inp.op.constants["weights"].set_numpy(np.tile(data, reps))
                inp.set_numpy(np.tile(data, reps))
            else:
                new_inp = ops.tile(inp, reps)
        new_inps.append(new_inp)
    return new_inps


class CodeGenCompass:
    """Compass IR codegen of Relax."""

    def __init__(self) -> None:
        super().__init__()
        self.var2val = {}
        self.var2tensor = {}
        self.const2tensor = {}
        self.name_count = {}
        self.is_quant = False
        self.unused_outputs = []

    def _get_valid_name(self, name):
        name_count = self.name_count.get(name, 0)
        name_count += 1
        self.name_count[name] = name_count
        return name + "_" + str(name_count)

    def _get_const(self, const: relax.Constant, quantization=None):
        if const in self.const2tensor and not self.is_quant:
            return self.const2tensor[const]
        data = const.data.numpy()
        data = data.astype("uint8") if data.dtype == "bool" else data
        data = data.astype("int32") if data.dtype == "int64" else data
        tensor = ops.constant(data, quantization=quantization)
        self.const2tensor[const] = tensor
        return tensor

    def _get_inputs(self, args):
        rets = []
        for arg in args:
            if isinstance(arg, relax.Constant):
                rets.append(self._get_const(arg))
            else:
                assert isinstance(arg, relax.Var)
                rets.append(self.var2tensor[arg])
        return rets

    def _get_sqo_info(self, op_name, args, func):
        """Get single quant op info.
        op_name: the single op name, e.g. mean
        args: args in call node which call the qnn func, e.g. gv(lv0)
        func: the callee qnn func, such as:
              def gv(param0, param1):
                lv = R.dequant(param0, ...)
                lv1 = R.mean(lv, ...)
                gv = R.quant(lv1, ...)
        return:
              inps: input tensors of caller, e.g. [self.var2tensor[lv0],]
              attrs: the attrs of inner op, e.g. mean.attrs
        """
        inner_call = _find_op_from_bindings(func.body.blocks[0].bindings, op_name)
        var2val = get_var2val(func)
        inner_call_args = inner_call.args[0].fields if op_name == "concat" else inner_call.args
        inner_call_indirect_args = []
        inp_quant_info = {}
        for arg in inner_call_args:
            if isinstance(arg, relax.Var) and arg in var2val:
                dequant = var2val[arg]
                inp = dequant.args[0]
                if isinstance(inp, relax.Constant):
                    inp_quant_info[inp] = _get_quant(*dequant.args[1:])
                inner_call_indirect_args.append(inp)
            else:
                inner_call_indirect_args.append(arg)
        params = list(func.params)
        inps = []
        for arg in inner_call_indirect_args:
            if isinstance(arg, relax.Constant):
                inps.append(self._get_const(arg, inp_quant_info[arg]))
            elif isinstance(arg, relax.Var):
                assert arg in params
                inps.append(self.var2tensor[args[params.index(arg)]])
            else:
                inps.append(arg)
        return inps, inner_call.attrs

    def gen(self, func):
        """Generate compass ir text and binary."""
        self.var2val = get_var2val(func)
        is_quant = CompassConfig.get().common["compat_quantized_model"] == "true"
        with Graph() as g:
            inp_quant_infos = None
            if is_quant:
                g.attrs["compat_quantized_model"] = True
                inp_quant_infos = func.attrs["quant_infos"]
                self.is_quant = True

            for i, param in enumerate(func.params):
                shape = list(param.struct_info.shape)
                dtype = _DTYPE_DICT[param.struct_info.dtype]
                inp = Tensor(param.name_hint, TensorShape(shape), dtype)
                if is_quant and len(inp_quant_infos) > 0:
                    inp.quantization = _get_quant(*inp_quant_infos[i])
                inp_node = Node(self._get_valid_name("input"), OpType.Input)
                inp_node.add_output(inp)
                g.add_node(inp_node)
                self.var2tensor[param] = inp

            if len(func.body.blocks) != 1:
                raise RuntimeError("Unsupport multiple blocks codegen now.")

            unary_ops = ["relu", "tanh", "sign", "sigmoid", "negative", "exp", "log", "erf", "sqrt"]
            unary_ops += ["silu", "logical_not", "abs"]

            binary_ops = ["logical_or", "logical_and", "logical_xor", "greater_equal", "less_equal"]
            binary_ops += ["equal", "less", "not_equal", "greater", "left_shift", "right_shift"]
            for bind in func.body.blocks[0].bindings:
                if not isinstance(bind, relax.VarBinding):
                    raise RuntimeError("Only support var binding codegen now.")
                if isinstance(bind.value, relax.Function):
                    # Skip composite function binding, codegen it in the next call op.
                    continue
                value = bind.value
                if isinstance(value, relax.Tuple):
                    self.var2tensor[bind.var] = self._get_inputs(value.fields)
                if isinstance(value, relax.TupleGetItem):
                    self.var2tensor[bind.var] = self.var2tensor[value.tuple_value][value.index]
                if not isinstance(value, relax.Call):
                    continue
                if isinstance(value.op, ir.Op):  # Single op: relax.xxx
                    op_name = value.op.name
                    # Delete prefix from: relax.nn.xxx, relax.image.xxx.
                    op_name = op_name.split(".")[-1] if "." in op_name else op_name
                    attrs = value.attrs
                    inps = [self.var2tensor[x] for x in value.args if isinstance(x, relax.Var)]
                    if op_name in ["conv2d", "conv2d_transpose"]:
                        ret = self._gen_single_conv2d_common(op_name, value)
                    elif op_name in unary_ops:
                        ret = getattr(ops, op_name)(*inps)
                    elif op_name == "prelu":
                        alpha = _create_tensor("alpha", value.args[1].data.numpy())
                        ret = ops.prelu(inps[0], alpha)
                    elif op_name in ["mean", "sum", "max", "min", "variance", "all", "any", "prod"]:
                        ret = getattr(ops, f"reduce_{op_name}")(inps[0], attrs.axis, attrs.keepdims)
                    elif op_name in ["add", "subtract", "multiply", "maximum", "minimum"]:
                        ops_func = getattr(ops, f"elementwise_{op_name[:3]}")
                        ret = ops_func(*self._get_inputs(value.args))
                    elif op_name in binary_ops:
                        ops_func = getattr(ops, op_name)
                        ret = ops_func(*self._get_inputs(value.args))
                    elif op_name == "cos":
                        ret = ops.cosine(inps[0])
                    elif op_name == "divide":
                        ret = ops.div(*self._get_inputs(value.args))
                    elif op_name == "reshape":
                        ret = ops.reshape(inps[0], [int(x) for x in value.args[1]])
                    elif op_name == "permute_dims":
                        ret = ops.transpose(inps[0], attrs.axes)
                    elif op_name == "max_pool2d":
                        ret = self._gen_max_pool2d(inps[0], attrs)
                    elif op_name == "avg_pool2d":
                        ret = self._gen_avg_pool2d(inps[0], attrs)
                    elif op_name == "matmul":
                        ret = ops.matmul(*self._get_inputs(value.args))
                    elif op_name == "pad":
                        pads = np.array(attrs.pad_width).reshape([-1, 2]).tolist()
                        ret = ops.pad(inps[0], pads, attrs.pad_mode, attrs.pad_value)
                    elif op_name == "clip":
                        ret = ops.clip(inps[0], value.args[1].value, value.args[2].value)
                    elif op_name == "concat":
                        ret = ops.concat(self._get_inputs(value.args[0].fields), attrs.axis)
                    elif op_name == "lrn":
                        ret = self._gen_lrn(inps[0], attrs)
                    elif op_name == "softmax":
                        ret = ops.softmax(inps[0], attrs.axis)
                    elif op_name == "log_softmax":
                        ret = ops.log_softmax(inps[0], attrs.axis)
                    elif op_name == "resize2d":
                        ret = self._gen_resize2d(inps[0], value.args[1].values, attrs)
                    elif op_name == "take":
                        ret = ops.gather(*self._get_inputs(value.args), attrs.axis)
                    elif op_name == "space_to_depth":
                        ret = ops.space_to_depth(inps[0], attrs.block_size)
                    elif op_name == "depth_to_space":
                        ret = ops.depth_to_space(inps[0], attrs.block_size, attrs.mode)
                    elif op_name == "strided_slice":
                        ret = self._gen_strided_slice(inps[0], value)
                    elif op_name == "split":
                        ret = self._gen_split(inps[0], attrs.indices_or_sections, attrs.axis)
                    elif op_name == "tile":
                        ret = ops.tile(inps[0], attrs.repeats)
                    elif op_name == "reverse_sequence":
                        ret = ops.reverse_sequence(*inps, attrs.seq_axis, attrs.batch_axis)
                    elif op_name == "power":
                        ret = ops.pow(*self._get_inputs(value.args))
                    elif op_name == "astype":
                        ret = self._gen_cast(inps[0], attrs)
                    elif op_name == "space_to_batch_nd":
                        ret = ops.space_to_batch(inps[0], attrs.block_shape, attrs.paddings)
                    elif op_name == "batch_to_space_nd":
                        ret = ops.batch_to_space(inps[0], attrs.block_shape, attrs.crops)
                    elif op_name == "scatter_elements":
                        reduction = "none" if attrs.reduction == "update" else attrs.reduction
                        ret = ops.scatter_elements(*inps, reduction, attrs.axis)
                    elif op_name == "fake_quant_with_min_max_vars":
                        ret = self._gen_fake_quant_min_max_vars(inps[0], attrs, g)
                    elif op_name == "ctc_greedy_decoder":
                        ret = ops.ctc_greedy_decoder(*inps, attrs.merge_repeated)
                    elif op_name == "requantize":
                        ret = self._gen_requantize(inps[0], value, attrs)
                    elif op_name == "dequantize":
                        quant_info = _get_quant(*value.args[1:3])
                        ret = ops.dequantize(inps[0], _DTYPE_DICT[attrs.out_dtype], quant_info)
                    elif op_name == "quantize":
                        out_dtype = _DTYPE_DICT[attrs.out_dtype]
                        quant_info = _get_quant(*value.args[1:3])
                        ret = ops.quantize(inps[0], out_dtype, "ROUND_TO_EVEN", quant_info)
                    elif op_name == "decode_box":
                        ret = self._gen_decode_box(inps, value, g)
                    elif op_name == "cps_nms":
                        ret = self._gen_cps_nms(inps, attrs)
                    elif op_name == "rms_norm":
                        weight = _create_tensor("weight", value.args[1].data.numpy())
                        ret = ops.rms_norm(inps[0], weight, None, attrs.axes, attrs.epsilon)
                    elif op_name == "channel_shuffle":
                        ret = ops.channel_shuffle(inps[0], attrs.group, attrs.axis, attrs.splits)
                        ret = ret[0] if len(ret) == 1 else ret
                    elif op_name == "one_hot":
                        values = [value.args[2].value.value, value.args[1].value.value]
                        ret = ops.one_hot(inps[0], attrs.depth, values, attrs.axis)
                    elif op_name in ["argmin", "argmax"]:
                        ret = getattr(ops, op_name)(inps[0], attrs.axis, attrs.keepdims)
                    elif op_name == "flip":
                        ret = self._gen_flip(inps[0], attrs)
                    elif op_name == "where":
                        ret = ops.where(*self._get_inputs(value.args))
                    elif op_name == "scatter_nd":
                        reduction = "none" if attrs.reduction == "update" else attrs.reduction
                        ret = ops.scatter_nd(*self._get_inputs(value.args), reduction)
                    elif op_name in CODEGEN_CUSTOM_OP_DICT:
                        ret = self._gen_custom_op(value, g, CODEGEN_CUSTOM_OP_DICT[op_name])
                    else:
                        raise RuntimeError(f"Unsupport op codegen: {op_name}")
                else:  # Composite func call: gv(lv)
                    assert isinstance(value.op, relax.Var)
                    compass_func = self.var2val[value.op]
                    func_name = compass_func.attrs["Composite"]
                    assert func_name.startswith("compass"), f"Unsupport codegen: {func_name}"
                    op_name = func_name[8:]
                    quant = None
                    if op_name.startswith("qnn."):
                        quant = _get_quant_info(compass_func)
                    # Delete prefix from: nn.xxx, image.xxx., qnn.xxx, qnn.nn.xxx
                    op_name = op_name.split(".")[-1] if "." in op_name else op_name
                    ret = None
                    q_out = quant["out"] if quant is not None else None
                    # This part is composite op with or not quant.
                    if op_name == "conv2d":
                        ret = self._gen_conv2d_composite(value.args, compass_func, quant)
                    elif op_name == "eltwise_relu":
                        ret = self._gen_eltwise_relu(value.args, compass_func, quant)
                    elif op_name == "instance_norm":
                        ret = self._gen_instance_norm(value.args, compass_func)
                    elif op_name == "layer_norm0":
                        ret = self._gen_layer_norm0(value.args, compass_func)
                    elif op_name == "layer_norm1":
                        ret = self._gen_layer_norm1(value.args, compass_func)
                    elif op_name == "batch_norm":
                        ret = self._gen_batch_norm(value.args, compass_func)
                    elif op_name == "batch_norm_single":
                        ret = self._gen_batch_norm_single(value.args, compass_func)
                    elif op_name == "matmul_add":
                        ret = self._gen_matmul_add(value.args, compass_func, quant)
                    elif op_name == "hard_swish":
                        ret = ops.hard_swish(self.var2tensor[value.args[0]], q_out)
                    elif op_name == "silu":
                        ret = ops.silu(self.var2tensor[value.args[0]], q_out)
                    elif op_name == "dense":
                        ret = self._gen_dense(value.args, compass_func)
                    elif op_name == "softplus":
                        ret = ops.softplus(self.var2tensor[value.args[0]])
                    elif op_name == "squared_diff":
                        ret = self._gen_squared_diff(value.args, quant)
                    elif op_name == "basic_lstm":
                        ret = self._gen_basic_lstm(value.args, compass_func)
                    if ret is not None:
                        self.var2tensor[bind.var] = ret
                        continue

                    # The following is simple op with quant.
                    inps, attrs = self._get_sqo_info(op_name, value.args, compass_func)
                    q_out = quant["out"]
                    if op_name in ["add", "subtract", "multiply"]:
                        ops_func = getattr(ops, f"elementwise_{op_name[:3]}")
                        ret = ops_func(*_broadcast_inps(inps), quantization=quant["out"])
                    elif op_name == "mean":
                        ret = ops.reduce_mean(inps[0], attrs.axis, attrs.keepdims, q_out)
                    elif op_name == "resize2d":
                        ret = self._gen_resize2d(inps[0], inps[1].values, attrs)
                    elif op_name in ["tanh", "sigmoid"]:
                        ret = getattr(ops, op_name)(inps[0], q_out)
                    elif op_name == "concat":
                        ret = ops.concat(inps, attrs.axis, q_out)
                    elif op_name == "avg_pool2d":
                        ret = self._gen_avg_pool2d(inps[0], attrs)
                        ret.quantization = q_out
                    elif op_name == "softmax":
                        ret = ops.softmax(inps[0], attrs.axis, quantization=q_out)
                    elif op_name == "max_pool2d":
                        ret = self._gen_max_pool2d(inps[0], attrs)
                        ret.quantization = q_out
                    elif op_name == "rsqrt":
                        ret = ops.rsqrt(inps[0], q_out)
                    elif op_name == "prelu":
                        alpha = inps[1].op.constants["weights"].np
                        alpha = _create_tensor("alpha", alpha, quant["alpha"])
                        ret = ops.prelu(inps[0], alpha, quantization=q_out)
                        g.remove_node(inps[1].op)
                    elif op_name == "matmul":
                        ret = ops.matmul(*inps, quantization=q_out)
                    else:
                        raise RuntimeError(f"Unsupport pattern codegen: {func_name}")

                self.var2tensor[bind.var] = ret

        # Remove unused output tensors.
        if len(self.unused_outputs) != 0:
            out_tensors = [x for x in g.output_tensors if x not in self.unused_outputs]
            g.output_tensors = ops.TensorList(out_tensors)

        # Adjust outputs' sequence align with relax's IR.
        func_out = func.body.body
        if isinstance(func_out.struct_info, relax.TupleStructInfo):
            func_out = self.var2val[func_out] if isinstance(func_out, relax.Var) else func_out
            out_tensors = [self.var2tensor[x] for x in func_out]
            out_tensors = ops.TensorList(out_tensors)
            if g.output_tensors != out_tensors:
                g.output_tensors = out_tensors

        return g.serialize()

    def gen2file(self, func, txt_path, bin_path):
        """Generate compass ir and save to file."""
        ir_txt, ir_bin = self.gen(func)
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        open(txt_path, "w", encoding="utf-8").write(ir_txt)
        os.makedirs(os.path.dirname(bin_path), exist_ok=True)
        open(bin_path, "wb").write(ir_bin)

    def _gen_single_conv2d_common(self, op_name, call):
        inp = self.var2tensor[call.args[0]]
        weight_data = call.args[1].data.numpy()
        weight = _create_tensor("weight", weight_data)
        bias_data = np.zeros(int(call.struct_info.shape[-1]), dtype="float32")
        bias = _create_tensor("bias", bias_data)

        attrs = call.attrs
        strides = [int(x) for x in attrs.strides]
        padding = [int(x) for x in attrs.padding]
        assert len(padding) == 4
        pad_top, pad_left, pad_bottom, pad_right = padding
        dilations = [int(x) for x in attrs.dilation]
        if op_name == "conv2d_transpose":
            output_padding = [int(x) for x in attrs.output_padding]
            pad_bottom -= output_padding[0]
            pad_right -= output_padding[1]
        pads = [pad_top, pad_bottom, pad_left, pad_right]
        is_dw_conv = attrs.groups == inp.shape[-1]

        if op_name == "conv2d":
            ops_func = ops.depthwise_conv2d if is_dw_conv else ops.conv2d
            conv_t = ops_func(inp, weight, bias, strides, pads, dilations)
        else:
            assert op_name == "conv2d_transpose"
            conv_t = ops.conv2d_transpose(inp, weight, bias, strides, pads, attrs.groups, dilations)
        return conv_t

    def _gen_conv2d_composite(self, inps, func, quant):
        inp = self.var2tensor[inps[0]]
        bindings = func.body.blocks[0].bindings
        var2val = get_var2val(func)
        conv = _find_op_from_bindings(bindings, "conv2d") or _find_op_from_bindings(
            bindings, "conv2d_transpose"
        )
        names = ["add", "relu", "leakyrelu", "clip"]
        add, *act = _find_op_from_bindings(bindings, names)
        act = [x for x in act if x]
        activation = None
        if len(act) != 0:
            act_name = act[0].op.name.split(".")[-1]
            if act_name == "relu":
                activation = "RELU"
            elif act_name == "clip":
                activation = "RELU6"
            else:
                assert act_name == "leakyrelu", f"Unknown act type {act_name}."
                activation = ops.Activation("LEAKYRELU", act[0].attrs.alpha)

        bias = None
        if quant:
            inp.quantization = quant["input"]
            weight_data = var2val[conv.args[1]].args[0].data.numpy()
            weight = _create_tensor("weight", weight_data, quant["weight"])
            if add:
                bias_data = var2val[add.args[1]].args[0].data.numpy().flatten()
                bias = _create_tensor("bias", bias_data, quant["bias"])
            else:
                zero_bias = np.zeros(int(conv.struct_info.shape[-1]), dtype="float32")
                bias = _create_tensor("bias", zero_bias)
        else:
            weight_data = conv.args[1].data.numpy()
            weight = _create_tensor("weight", weight_data)
            if add:
                bias_data = add.args[1].data.numpy().flatten()
            else:
                bias_data = np.zeros(int(conv.struct_info.shape[-1]), dtype="float32")
            bias = _create_tensor("bias", bias_data)

        attrs = conv.attrs
        strides = [int(x) for x in attrs.strides]
        padding = [int(x) for x in attrs.padding]
        assert len(padding) == 4
        pad_top, pad_left, pad_bottom, pad_right = padding
        dilations = [int(x) for x in attrs.dilation]
        op_name = str(conv.op.name).split(".")[-1]
        if op_name == "conv2d_transpose":
            output_padding = [int(x) for x in attrs.output_padding]
            pad_bottom -= output_padding[0]
            pad_right -= output_padding[1]
        pads = [pad_top, pad_bottom, pad_left, pad_right]
        is_dw_conv = attrs.groups == inp.shape[-1]
        out_quant = quant["out"] if quant else None

        if op_name == "conv2d":
            ops_func = ops.depthwise_conv2d if is_dw_conv else ops.conv2d
            conv_t = ops_func(inp, weight, bias, strides, pads, dilations, activation, out_quant)
        else:
            assert op_name == "conv2d_transpose"
            conv_t = ops.conv2d_transpose(
                inp, weight, bias, strides, pads, attrs.groups, dilations, activation, out_quant
            )
        return conv_t

    def _gen_eltwise_relu(self, call_args, func, quant=None):
        var2val = get_var2val(func)
        bindings = func.body.blocks[0].bindings
        relu = _find_op_from_bindings(bindings, "relu")
        if quant:
            relu_dq = var2val[relu.args[0]]
            elem_q = var2val[relu_dq.args[0]]
            elem = var2val[elem_q.args[0]]
            elem_name = elem.op.name.split(".")[1]
            inps, _ = self._get_sqo_info(elem_name, call_args, func)
            q_out = quant["out"]
        else:
            elem = var2val[relu.args[0]]
            elem_name = elem.op.name.split(".")[1]
            params = list(func.params)
            inps = []
            for arg in elem.args:
                if isinstance(arg, relax.Constant):
                    inps.append(self._get_const(arg))
                else:
                    assert isinstance(arg, relax.Var) and arg in params
                    inps.append(self.var2tensor[call_args[params.index(arg)]])
            q_out = None

        ops_func = getattr(ops, f"elementwise_{elem_name[:3]}")
        return ops_func(*_broadcast_inps(inps), "RELU", quantization=q_out)

    def _gen_max_pool2d(self, inp, attrs):
        kernel = [int(x) for x in attrs.pool_size]
        strides = [int(x) for x in attrs.strides]
        dilation = [int(x) for x in attrs.dilation]
        padding = [int(x) for x in attrs.padding]
        if len(padding) == 4:
            # Convert from relax [top, left, bottom, right]
            # to gb's: [top, bottom, left, right].
            padding = [padding[idx] for idx in [0, 2, 1, 3]]

        max_pool2d_tensor = ops.max_pool(inp, kernel, strides, dilation, padding, attrs.ceil_mode)
        return max_pool2d_tensor

    def _gen_avg_pool2d(self, inp, attrs):
        kernel = [int(x) for x in attrs.pool_size]
        strides = [int(x) for x in attrs.strides]
        dilation = [int(x) for x in attrs.dilation]
        padding = [int(x) for x in attrs.padding]
        if len(padding) == 4:
            # Convert from relax [top, left, bottom, right]
            # to gb's: [top, bottom, left, right].
            padding = [padding[idx] for idx in [0, 2, 1, 3]]

        return ops.avg_pool(inp, kernel, strides, dilation, padding, attrs.ceil_mode)

    def _gen_matmul_add(self, inps, func, quant):
        inp = self.var2tensor[inps[0]]
        var2val = get_var2val(func)
        bindings = func.body.blocks[0].bindings
        matmul, add = _find_op_from_bindings(bindings, ["matmul", "add"])
        weight = matmul.args[1]
        bias = add.args[1]

        w_quant, b_quant, out_quant = None, None, None
        if quant:
            weight = var2val[weight].args[0]
            bias = var2val[bias].args[0]
            w_quant, b_quant, out_quant = quant["weight"], quant["bias"], quant["out"]

        weight_data = weight.data.numpy()
        weight_data = np.transpose(weight_data, [1, 0])
        weight = _create_tensor("weight", weight_data, w_quant)
        bias_data = bias.data.numpy()
        bias = _create_tensor("bias", bias_data, b_quant)

        act = None
        if len(bindings) == 3:
            act = str(bindings[2].value.op.name).split(".")[-1]
            assert act in ["relu"]

        out = ops.fully_connected(inp, weight, bias, activation=act, quantization=out_quant)
        return out

    def _gen_instance_norm(self, inps, func):
        inp = self.var2tensor[inps[0]]
        bindings = func.body.blocks[0].bindings
        gamma_data = bindings[-2].value.args[1].data.numpy().flatten()
        gamma = _create_tensor("gamma", gamma_data)
        beta_data = bindings[-1].value.args[1].data.numpy().flatten()
        beta = _create_tensor("beta", beta_data)
        return ops.instance_norm(inp, gamma, beta)

    def _gen_layer_norm0(self, inps, func):
        inp = self.var2tensor[inps[0]]
        bindings = func.body.blocks[0].bindings
        # The following index in bindings refer to pattern table.
        axis = bindings[0].value.attrs.axis[0]
        epsilon = bindings[4].value.args[1].data.numpy().item()
        gamma_data = bindings[-2].value.args[1].data.numpy().flatten()
        gamma = _create_tensor("gamma", gamma_data)
        beta_data = bindings[-1].value.args[1].data.numpy().flatten()
        beta = _create_tensor("beta", beta_data)
        return ops.layer_norm(inp, gamma, beta, axis, epsilon)

    def _gen_layer_norm1(self, inps, func):
        inp = self.var2tensor[inps[0]]
        bindings = func.body.blocks[0].bindings
        # The following index in bindings refer to pattern table.
        axis = int(bindings[0].value.attrs.axis[0])
        epsilon = bindings[3].value.args[1].data.numpy().item()

        def _tile_data_if_needed(inp_data, target_length):
            if inp_data.size != target_length:
                inp_data = np.tile(inp_data, target_length // inp_data.size)
            return inp_data

        out_shape = [int(x) for x in bindings[-1].value.struct_info.shape]
        gamma_data = np.ones(1, "float32")
        beta_data = np.zeros(1, "float32")
        if len(bindings) > 6:
            gamma_data = bindings[6].value.args[1].data.numpy().flatten()
            beta_data = bindings[-1].value.args[1].data.numpy().flatten()
        gamma_data = _tile_data_if_needed(gamma_data, out_shape[axis])
        beta_data = _tile_data_if_needed(beta_data, out_shape[axis])
        gamma = _create_tensor("gamma", gamma_data)
        beta = _create_tensor("beta", beta_data)
        return ops.layer_norm(inp, gamma, beta, axis, epsilon)

    def _gen_batch_norm(self, inps, func):
        inp = self.var2tensor[inps[0]]
        bindings = func.body.blocks[0].bindings
        names = ["subtract", "divide", "multiply", "add"]
        sub, div, mul, add = _find_op_from_bindings(bindings, names)
        _, sub_const = unpack_commutative_args(sub)
        _, div_const = unpack_commutative_args(div)
        _, mul_const = unpack_commutative_args(mul)
        _, add_const = unpack_commutative_args(add)
        sub_data = sub_const.data.numpy().flatten()
        div_data = div_const.data.numpy().flatten()
        mul_data = mul_const.data.numpy().flatten()
        add_data = add_const.data.numpy().flatten()

        scale = 1 / div_data
        scale = scale * mul_data
        neg_mean = -sub_data
        shift = neg_mean * scale
        shift = shift + add_data
        weight = _create_tensor("weight", scale)
        bias = _create_tensor("bias", shift)
        axis = len(inps[0].struct_info.shape) - 1

        return ops.batch_norm(inp, weight, bias, axis)

    def _gen_batch_norm_single(self, inps, func):
        inp = self.var2tensor[inps[0]]
        c = int(inps[0].struct_info.shape[-1])
        call = func.body.blocks[0].bindings[0].value
        _, const = unpack_commutative_args(call)
        const = const.data.numpy()
        op_name = call.op.name[6:]
        if op_name in ["add", "subtract"]:
            mul_const = np.ones([c], dtype=np.float32)
            add_const = const
            if op_name == "subtract":
                add_const = -const
        else:
            mul_const = const
            add_const = np.zeros([c], dtype=np.float32)
            if op_name == "divide":
                mul_const = 1 / mul_const
        add_const_shape = add_const.shape
        mul_const_shape = mul_const.shape
        if len(add_const_shape) == 0 or add_const_shape[-1] == 1:
            add_const = np.tile(add_const.reshape([-1]), [c])
        if len(mul_const_shape) == 0 or mul_const_shape[-1] == 1:
            mul_const = np.tile(mul_const.reshape([-1]), [c])
        weight = _create_tensor("weight", mul_const.reshape(-1))
        bias = _create_tensor("bias", add_const.reshape(-1))
        axis = len(inps[0].struct_info.shape) - 1

        return ops.batch_norm(inp, weight, bias, axis)

    def _gen_lrn(self, inp, attrs):
        # Here the layout is implicitly NHWC.
        method = "WITHIN_CHANNEL" if attrs.axis in [1, 2] else "ACROSS_CHANNELS"
        bias = attrs.bias
        return ops.lrn(inp, attrs.size, method, attrs.alpha, attrs.beta, bias)

    def _gen_resize2d(self, inp, size, attrs):
        method = "nearest"
        if attrs.method != "nearest_neighbor":
            method = "bilinear"
        mode = attrs.coordinate_transformation_mode.upper().replace("_", "")
        nearest_mode = attrs.rounding_method.upper().replace("_", "")
        # default value of nearest mode: tvm 'ROUND' vs compass 'ROUNDPREFERFLOOR'
        nearest_mode = "ROUNDPREFERFLOOR" if nearest_mode == "ROUND" else nearest_mode
        return ops.resize(inp, size, [], method, mode, nearest_mode)

    def _gen_strided_slice(self, inp, call):
        dims = len(inp.shape)
        axes, begins, ends = [[int(x.value) for x in y] for y in call.args[1:4]]
        inp_num = len(call.args)
        strides = [1] * len(begins) if inp_num == 4 else [int(x.value) for x in call.args[4]]

        new_begins = [0] * dims
        new_ends = [np.iinfo(np.int32).max] * dims
        new_strides = [1] * dims
        for i, axis in enumerate(axes):
            new_begins[int(axis)] = begins[i]
            new_ends[int(axis)] = ends[i]
            new_strides[int(axis)] = strides[i]

        input_shape = [int(x) for x in call.args[0].struct_info.shape]

        def clamp(value, idx):
            max_value = input_shape[idx]
            min_value = -input_shape[idx] - 1
            return max(min(int(value), max_value), min_value)

        new_begins = [clamp(val, idx) for idx, val in enumerate(new_begins)]
        new_ends = [clamp(val, idx) for idx, val in enumerate(new_ends)]
        return ops.strided_slice(inp, new_begins, new_ends, new_strides)

    def _gen_split(self, inp, indices_or_sections, axis):
        splits = indices_or_sections
        if isinstance(splits, tvm.ffi.Array):
            splits = [int(x) for x in splits]
            splits = np.diff([0, *splits, inp.shape[axis]]).tolist()
        return ops.split(inp, splits, axis)

    def _gen_dense(self, inps, func):
        inp = self.var2tensor[inps[0]]
        matmul = func.body.blocks[0].bindings[0].value
        _, weight = unpack_commutative_args(matmul)
        call_sinfo = matmul.struct_info
        weight_data = weight.data.numpy()
        weight_data = np.transpose(weight_data, [1, 0])
        weight = _create_tensor("weight", weight_data)
        bias_data = np.zeros([int(call_sinfo.shape[-1])], call_sinfo.dtype)
        bias = _create_tensor("bias", bias_data)

        return ops.fully_connected(inp, weight, bias)

    def _gen_cast(self, inp, attrs):
        dtype_mapping = {
            "bool": "uint8",
            "int64": "int32",
        }
        dtype = dtype_mapping[attrs.dtype] if attrs.dtype in dtype_mapping else attrs.dtype
        return ops.cast(inp, _DTYPE_DICT[dtype], inp.dtype == "int32")

    def _gen_fake_quant_min_max_vars(self, inp, attrs, graph):
        name = self._get_valid_name("fake_quant_with_min_max_vars")
        output = Tensor(name, inp.shape, inp.dtype)
        node = Node(name, OpType.FakeQuantWithMinMaxVars)
        node.add_input(inp)
        node.add_output(output)
        node.params["max"] = attrs.maximum
        node.params["min"] = attrs.minimum
        node.params["narrow_range"] = attrs.narrow_range
        node.params["num_bits"] = attrs.num_bits
        graph.add_node(node)
        return output

    def _gen_decode_box(self, inps, call, graph):
        name = self._get_valid_name("decode_box")
        node = Node(name, OpType.DecodeBox)
        constants = [x.data.numpy() for x in call.args[2:]]
        node.add_input(inps[0])
        node.add_input(inps[1])
        node.constants["ycenter"] = _create_tensor("ycenter", constants[0])
        node.constants["xcenter"] = _create_tensor("xcenter", constants[1])
        node.constants["ha"] = _create_tensor("ha", constants[2])
        node.constants["wa"] = _create_tensor("wa", constants[3])
        node.params["image_width"] = int(call.attrs.image_width)
        node.params["image_height"] = int(call.attrs.image_height)
        node.params["max_box_num"] = int(call.attrs.max_box_num)
        node.params["class_num"] = int(call.attrs.class_num)
        node.params["score_threshold"] = float(call.attrs.score_threshold)
        outputs = []
        for sinfo in call.struct_info.fields:
            out_shape = TensorShape(list(sinfo.shape))
            out_dtype = _DTYPE_DICT[sinfo.dtype]
            out = Tensor(self._get_valid_name(name + "_out"), out_shape, out_dtype)
            node.add_output(out)
            outputs.append(out)
        graph.add_node(node)
        return outputs

    def _gen_cps_nms(self, inps, attrs):
        args = [*inps, [attrs.image_width, attrs.image_height], attrs.method]
        args += [attrs.max_output_size, attrs.score_threshold, attrs.iou_threshold]
        args += [attrs.soft_nms_sigma, attrs.center_point_box]
        return ops.nms(*args)

    def _gen_custom_op(self, call, graph, type_name):
        if hasattr(OpType, type_name):
            op_type = getattr(OpType, type_name)
        else:
            op_type = register_optype(type_name)
        name = self._get_valid_name(type_name)
        node = Node(name, op_type)

        inps = self._get_inputs(call.args)
        for inp in inps:
            node.add_input(inp)
        out_sinfo = call.struct_info
        out_tensors = []
        if isinstance(out_sinfo, relax.TensorStructInfo):
            out_shape = TensorShape(list(out_sinfo.shape))
            out_dtype = _DTYPE_DICT[out_sinfo.dtype]
            out = Tensor(self._get_valid_name(name + "_out"), out_shape, out_dtype)
            node.add_output(out)
            out_tensors.append(out)
        else:
            assert isinstance(out_sinfo, relax.TupleStructInfo)
            for sinfo in out_sinfo.fields:
                out_shape = TensorShape(list(sinfo.shape))
                out_dtype = _DTYPE_DICT[sinfo.dtype]
                out = Tensor(self._get_valid_name(name + "_out"), out_shape, out_dtype)
                node.add_output(out)
                out_tensors.append(out)
        for k, v in call.attrs.items():
            node.params[k] = v
        graph.add_node(node)
        return out_tensors[0] if len(out_tensors) == 1 else out_tensors

    def _gen_requantize(self, inp, call, attrs):
        ignore_scale_zp = inp.dtype == Dtype.INT32
        out = ops.cast(inp, _DTYPE_DICT[attrs.out_dtype], ignore_scale_zp)
        inp.quantization = _get_quant(*call.args[1:3])
        out.quantization = _get_quant(*call.args[3:])
        return out

    def _gen_squared_diff(self, inps, quant):
        inp0 = self.var2tensor[inps[0]]
        inp1 = self.var2tensor[inps[1]]
        return ops.squared_difference(inp0, inp1, quant["out"])

    def _gen_flip(self, inp, attrs):
        inp_shape = inp.shape
        dim = len(inp_shape)
        axis = attrs.axis
        axis = dim + axis if axis < 0 else axis
        if axis in (0, 1):
            time_axis = axis
            batch = inp_shape[1 - time_axis]
            seq_len = inp_shape[time_axis]
            seq_lengths = ops.constant(np.array([seq_len] * batch, "int32"))
            return ops.reverse_sequence(inp, seq_lengths, time_axis, 1 - time_axis)
        else:
            pre_perm = [axis] + [idx for idx in range(dim) if idx != axis]
            pre_transpose = ops.transpose(inp, pre_perm)
            time_axis = 0
            batch = inp_shape[0]
            seq_len = inp_shape[axis]
            seq_lengths = ops.constant(np.array([seq_len] * batch, "int32"))
            reverse_seq = ops.reverse_sequence(pre_transpose, seq_lengths, time_axis, 1)
            post_perm = [pre_perm.index(i) for i in range(len(pre_perm))]
            return ops.transpose(reverse_seq, post_perm)

    def _gen_basic_lstm(self, inps, func):
        inps = self._get_inputs(inps)
        var2val = get_var2val(func)
        bindings = func.body.blocks[0].bindings
        matmul0, matmul1 = _find_all_same_op_from_bindings(bindings, "matmul")
        add0, add1, _ = _find_all_same_op_from_bindings(bindings, "add")

        add0_dq = var2val[add0.args[0]]
        matmul_q_find_by_add0 = var2val[add0_dq.args[0]]
        matmul_find_by_add0 = var2val[matmul_q_find_by_add0.args[0]]
        if matmul_find_by_add0 == matmul1:
            matmul0, matmul1 = matmul1, matmul0
            inps[0], inps[1] = inps[1], inps[0]

        weight_dq = var2val[matmul0.args[1]]
        weight = weight_dq.args[0].data.numpy()
        weight = np.transpose(weight, [1, 0])
        weight_scale = weight_dq.args[1].data.numpy()
        weight_zp = weight_dq.args[2].data.numpy()
        recurrent_weight_dq = var2val[matmul1.args[1]]
        recurrent_weight = recurrent_weight_dq.args[0].data.numpy()
        recurrent_weight = np.transpose(recurrent_weight, [1, 0])
        recurrent_weight_scale = recurrent_weight_dq.args[1].data.numpy()
        recurrent_weight_zp = recurrent_weight_dq.args[2].data.numpy()
        bias_dq = var2val[add1.args[1]]
        bias = bias_dq.args[0].data.numpy()

        i_w, f_w, c_w, o_w = np.split(weight, 4, axis=0)
        i_r, f_r, c_r, o_r = np.split(recurrent_weight, 4, axis=0)
        i_wb, f_wb, c_wb, o_wb = np.split(bias, 4, axis=0)
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
        weights_scale = np.array([weight_scale, recurrent_weight_scale], "float32")
        weights_zp = np.array([weight_zp, recurrent_weight_zp], "int32")
        weight_t = _create_tensor("weights", weights, Quantization(weights_scale, weights_zp))
        bias_t = _create_tensor("bias", biases, _get_quant(*bias_dq.args[1:]))

        def _get_arg_q(node_q):
            """Find arg quant: node_q -> node -> arg_dq -> arg_q"""
            node = var2val[node_q.args[0]]
            outputs = []
            for x in node.args:
                arg_dq = var2val[x]
                if arg_dq.args[0] in var2val:
                    outputs.append(var2val[arg_dq.args[0]])
                else:
                    outputs.append(arg_dq.args[0])
            return outputs[0] if len(outputs) == 1 else outputs

        mul_hout = _find_all_same_op_from_bindings(bindings, "quantize")[-1]
        sigmoid_sp3, tanh_c = _get_arg_q(mul_hout)
        add_cout = _get_arg_q(tanh_c)
        mul_sp2, mul_sp01 = _get_arg_q(add_cout)
        sigmoid_sp0, tanh_sp2 = _get_arg_q(mul_sp01)
        sigmoid_sp1, _ = _get_arg_q(mul_sp2)
        split = _find_op_from_bindings(bindings, "split")
        add_bias = var2val[split.args[0]]
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
            assert node.op.name == "relax.quantize", "Expect quantize op here."
            activations_scale.append(node.args[1].data.numpy())
            activations_zp.append(node.args[2].data.numpy())

        out_quant = _get_quant(*mul_hout.args[1:])
        cell_quant = _get_quant(*add_cout.args[1:])
        acts = ["Sigmoid", "Tanh", "Tanh"]
        ret = ops.basic_lstm(*inps, weight_t, bias_t, acts, cell_quant, out_quant, "H_C")
        lstm_node = ret[0].op
        name = "activations_scale"
        lstm_node.constants[name] = _create_tensor(name, np.array(activations_scale))
        name = "activations_zp"
        lstm_node.constants[name] = _create_tensor(name, np.array(activations_zp))

        func_out = func.body.body
        if isinstance(func_out, relax.Tuple):
            return ret
        assert isinstance(func_out, relax.Var), f"Unknown output {func_out} in func: {func}"
        for bind in bindings:
            if bind.var == func_out:
                out_quant = bind.value
                out_op = var2val[out_quant.args[0]]
                if out_op.op.name == "relax.multiply":
                    self.unused_outputs.append(ret[1])
                    return ret[0]
                else:
                    self.unused_outputs.append(ret[0])
                    return ret[1]
