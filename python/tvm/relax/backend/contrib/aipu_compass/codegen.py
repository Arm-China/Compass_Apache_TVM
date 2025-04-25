# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""AIPU Compass IR codegen of Relax."""
import os
import numpy as np
from tvm import relax, ir
from tvm.relax.analysis import get_var2val
from AIPUBuilder import ops
from AIPUBuilder.core import Tensor, TensorShape, Dtype, Graph
from AIPUBuilder._C._core import Quantization
from .config import AipuCompassConfig


_DTYPE_DICT = {
    bool: Dtype.BOOL,
    "float16": Dtype.FP16,
    "float32": Dtype.FP32,
    "float64": Dtype.FP64,
    "int16": Dtype.INT16,
    "int32": Dtype.INT32,
    "int64": Dtype.INT64,
    "int8": Dtype.INT8,
    "uint8": Dtype.UINT8,
}


def _create_tensor(name, data, quantization=None):
    tensor = Tensor(name)
    tensor.set_numpy(data)
    if quantization:
        tensor.quantization = quantization
    return tensor


def _find_op_from_bindings(bindings, names):
    names = names if isinstance(names, (tuple, list)) else [names]
    names = ["relax." + n for n in names]
    ret = [None] * len(names)
    for bind in bindings:
        value = bind.value
        op_name = value.op.name
        if op_name in names:
            ret[names.index(op_name)] = value
    return ret[0] if len(ret) == 1 else ret


def _get_quant(scale, zero_point):
    scale = scale.data.numpy()
    zero_point = zero_point.data.numpy()
    scale = float(scale) if len(scale.shape) == 0 else scale
    zero_point = int(zero_point) if len(zero_point.shape) == 0 else zero_point
    out = Quantization(scale, zero_point)
    out.unquantifiable = True
    return out


def _is_dq_const(x):
    if not x.op == ir.Op.get("relax.dequantize"):
        return False
    return isinstance(x.args[0], relax.Constant)


def _get_quant_info(op_name, func):
    quant_info = {}
    var2val = get_var2val(func)
    bindings = func.body.blocks[0].bindings
    for bind in bindings:
        value = bind.value
        op_name = value.op.name
        if op_name == "relax.nn.conv2d":
            weight_dq = var2val[value.args[1]]
            quant_info["weight"] = _get_quant(*weight_dq.args[1:])
        if op_name == "relax.add":
            bias_dq = var2val[value.args[1]]
            if _is_dq_const(bias_dq):
                quant_info["bias"] = _get_quant(*bias_dq.args[1:])
        if op_name == "relax.matmul":
            weight_dq = var2val[value.args[1]]
            quant_info["weight"] = _get_quant(*weight_dq.args[1:])
    quant_info["out"] = _get_quant(*bindings[-1].value.args[1:])
    return quant_info


class CodeGenAipuCompass:
    """AIPU Compass IR codegen of Relax."""

    def __init__(self) -> None:
        super().__init__()
        self.var2val = {}
        self.var2tensor = {}

    def gen2file(self, func, txt_path, bin_path):
        """Generate compass ir text and binary."""
        self.var2val = get_var2val(func)
        is_quant = AipuCompassConfig.get().common["compat_quantized_model"] == "true"
        with Graph() as g:
            inp_quant_infos = None
            if is_quant:
                g.attrs["compat_quantized_model"] = True
                inp_quant_infos = func.attrs["quant_infos"]

            for i, param in enumerate(func.params):
                shape = list(param.struct_info.shape)
                dtype = _DTYPE_DICT[param.struct_info.dtype]
                inp = Tensor(param.name_hint, TensorShape(shape), dtype)
                if is_quant:
                    inp.quantization = _get_quant(*inp_quant_infos[i])
                self.var2tensor[param] = inp

            for bind in func.body.blocks[0].bindings:
                if not isinstance(bind, relax.VarBinding):
                    continue
                if isinstance(bind.value, relax.Function):
                    continue
                value = bind.value
                if not isinstance(value, relax.Call):
                    continue

                compass_func = self.var2val[value.op]
                func_name = compass_func.attrs["Composite"]
                assert func_name.startswith("aipu_compass."), f"Unsupport op codegen: {func_name}"
                op_name = func_name[13:]
                quant = None
                if op_name.startswith("qnn."):
                    op_name = op_name[4:]
                    quant = _get_quant_info(op_name, compass_func)

                if op_name == "conv2d":
                    self._gen_conv2d(bind.var, value.args, compass_func, quant)
                elif op_name in ["relu", "tanh", "sign"]:
                    self._gen_simple_op(bind.var, value.args, op_name)
                elif op_name == "eltwise_relu":
                    self._gen_elemtwise_relu(bind.var, value.args, compass_func)
                elif op_name == "add":
                    self._gen_add(bind.var, value.args, compass_func, quant)
                elif op_name == "subtract":
                    self._gen_subtract(bind.var, value.args, compass_func)
                elif op_name == "multiply":
                    self._gen_multiply(bind.var, value.args, compass_func)
                elif op_name == "divide":
                    self._gen_divide(bind.var, value.args, compass_func)
                elif op_name == "reshape":
                    self._gen_reshape(bind.var, value.args, compass_func)
                elif op_name == "transpose":
                    self._gen_transpose(bind.var, value.args, compass_func)
                elif op_name == "max_pool2d":
                    self._gen_max_pool2d(bind.var, value.args, compass_func)
                elif op_name == "mean":
                    self._gen_mean(bind.var, value.args, compass_func)
                elif op_name == "matmul_add":
                    self._gen_matmul_add(bind.var, value.args, compass_func, quant)
                elif op_name == "matmul":
                    self._gen_matmul(bind.var, value.args, compass_func)
                elif op_name == "instance_norm":
                    self._gen_instance_norm(bind.var, value.args, compass_func)
                elif op_name == "pad":
                    self._gen_pad(bind.var, value.args, compass_func)
                elif op_name == "clip":
                    self._gen_clip(bind.var, value.args, compass_func)
                elif op_name == "concat":
                    self._gen_concat(bind.var, value.args, compass_func)
                elif op_name == "tuple":
                    self._gen_tuple(bind.var, value.args)
                else:
                    raise RuntimeError(f"Unsupport op codegen: {func_name}")

        ir_txt, ir_bin = g.serialize()
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        open(txt_path, "w", encoding="utf-8").write(ir_txt)
        os.makedirs(os.path.dirname(bin_path), exist_ok=True)
        open(bin_path, "wb").write(ir_bin)

    def _gen_conv2d(self, var, inps, func, quant):
        inp = self.var2tensor[inps[0]]
        bindings = func.body.blocks[0].bindings
        var2val = get_var2val(func)
        op2act = {
            "relax.nn.relu": "RELU",
            "relax.nn.leakyrelu": "LEAKYRELU",
            "relax.clip": "RELU6",
        }
        conv = _find_op_from_bindings(bindings, "nn.conv2d") or _find_op_from_bindings(
            bindings, "nn.conv2d_transpose"
        )
        names = ["add", "nn.relu", "nn.leakyrelu", "clip"]
        add, *act = _find_op_from_bindings(bindings, names)
        act = [x.op.name for x in act if x]
        activation = op2act[act[0]] if len(act) > 0 else None

        bias = None
        if quant:
            weight_data = var2val[conv.args[1]].args[0].data.numpy()
            weight = _create_tensor("weight", weight_data, quant["weight"])
            if add:
                bias_data = var2val[add.args[1]].args[0].data.numpy().flatten()
                bias = _create_tensor("bias", bias_data, quant["bias"])
        else:
            weight_data = conv.args[1].data.numpy()
            weight = _create_tensor("weight", weight_data)
            if add:
                bias_data = add.args[1].data.numpy().flatten()
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
        self.var2tensor[var] = conv_t

    def _gen_simple_op(self, var, inps, name):
        inp = self.var2tensor[inps[0]]
        self.var2tensor[var] = getattr(ops, name)(inp)

    def _gen_elemtwise_relu(self, var, inps, func):
        inp = self.var2tensor[inps[0]]
        elem = func.body.blocks[0].bindings[0].value
        if elem.args[0] == elem.args[1]:
            inp1 = inp
        elif isinstance(elem.args[1], relax.Constant):
            const_data = elem.args[1].data.numpy()
            inp1 = ops.constant(const_data)
        else:
            inp1 = self.var2tensor[inps[1]]

        elem_name = elem.op.name.split(".")[1][:3]
        ops_func = getattr(ops, elem_name)
        self.var2tensor[var] = ops_func(inp, inp1, "RELU")

    def _gen_add(self, var, inps, func, quant):
        var2val = get_var2val(func)
        params = list(func.params)
        add = _find_op_from_bindings(func.body.blocks[0].bindings, "add")
        lhs, rhs = add.args
        if quant:
            lhs = var2val[lhs].args[0]
            rhs = var2val[rhs].args[0]
        assert lhs in params
        inp0 = self.var2tensor[inps[params.index(lhs)]]
        if lhs == rhs:  # func(x): return R.add(x, x)
            inp1 = inp0
        elif isinstance(rhs, relax.Constant):
            const_data = rhs.data.numpy()
            inp1 = ops.constant(const_data)
        else:
            assert rhs in params
            inp1 = self.var2tensor[inps[params.index(rhs)]]
        out_quant = quant["out"] if quant else None

        out = ops.add(inp0, inp1, quantization=out_quant)
        self.var2tensor[var] = out

    def _gen_subtract(self, var, inps, func):
        inp = self.var2tensor[inps[0]]
        sub = func.body.blocks[0].bindings[0].value
        if sub.args[0] == sub.args[1]:
            inp1 = inp
        elif isinstance(sub.args[1], relax.Constant):
            const_data = sub.args[1].data.numpy()
            inp1 = ops.constant(const_data)
        else:
            inp1 = self.var2tensor[inps[1]]

        self.var2tensor[var] = ops.sub(inp, inp1)

    def _gen_multiply(self, var, inps, func):
        inp = self.var2tensor[inps[0]]
        mul = func.body.blocks[0].bindings[0].value
        if mul.args[0] == mul.args[1]:
            inp1 = inp
        elif isinstance(mul.args[1], relax.Constant):
            const_data = mul.args[1].data.numpy()
            inp1 = ops.constant(const_data)
        else:
            inp1 = self.var2tensor[inps[1]]

        self.var2tensor[var] = ops.mul(inp, inp1)

    def _gen_divide(self, var, inps, func):
        inp = self.var2tensor[inps[0]]
        div = func.body.blocks[0].bindings[0].value
        if div.args[0] == div.args[1]:
            inp1 = inp
        elif isinstance(div.args[1], relax.Constant):
            const_data = div.args[1].data.numpy()
            inp1 = ops.constant(const_data)
        else:
            inp1 = self.var2tensor[inps[1]]

        self.var2tensor[var] = ops.div(inp, inp1)

    def _gen_reshape(self, var, inps, func):
        inp = self.var2tensor[inps[0]]
        new_shape = func.body.blocks[0].bindings[0].value.args[1]
        new_shape = [int(x) for x in new_shape]
        self.var2tensor[var] = ops.reshape(inp, new_shape)

    def _gen_transpose(self, var, inps, func):
        inp = self.var2tensor[inps[0]]
        transpose = func.body.blocks[0].bindings[0].value
        axes = [int(x) for x in transpose.attrs.axes]
        self.var2tensor[var] = ops.transpose(inp, axes)

    def _gen_max_pool2d(self, var, inps, func):
        inp = self.var2tensor[inps[0]]
        max_pool2d = _find_op_from_bindings(func.body.blocks[0].bindings, "nn.max_pool2d")

        attrs = max_pool2d.attrs
        kernel = [int(x) for x in attrs.pool_size]
        strides = [int(x) for x in attrs.strides]
        dilation = [int(x) for x in attrs.dilation]
        padding = [int(x) for x in attrs.padding]
        max_pool2d_tensor = ops.max_pool(inp, kernel, strides, dilation, padding, attrs.ceil_mode)
        self.var2tensor[var] = max_pool2d_tensor

    def _gen_mean(self, var, inps, func):
        inp = self.var2tensor[inps[0]]
        mean = _find_op_from_bindings(func.body.blocks[0].bindings, "mean")
        attrs = mean.attrs
        axis = [int(x) for x in attrs.axis]
        self.var2tensor[var] = ops.mean(inp, axis, attrs.keepdims)

    def _gen_matmul_add(self, var, inps, func, quant):
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

        out = ops.fully_connected(inp, weight, bias, quantization=out_quant)
        self.var2tensor[var] = out

    def _gen_matmul(self, var, inps, func):
        inp = self.var2tensor[inps[0]]
        matmul = _find_op_from_bindings(func.body.blocks[0].bindings, "matmul")
        if isinstance(matmul.args[1], relax.Constant):
            const_data = matmul.args[1].data.numpy()
            inp1 = ops.constant(const_data)
        elif matmul.args[0] == matmul.args[1]:
            inp1 = inp
        else:
            inp1 = self.var2tensor[inps[1]]

        self.var2tensor[var] = ops.matmul(inp, inp1)

    def _gen_instance_norm(self, var, inps, func):
        inp = self.var2tensor[inps[0]]
        bindings = func.body.blocks[0].bindings
        gamma_data = bindings[-2].value.args[1].data.numpy().flatten()
        gamma = Tensor("gamma")
        gamma.set_numpy(gamma_data)
        beta_data = bindings[-1].value.args[1].data.numpy().flatten()
        beta = Tensor("beta")
        beta.set_numpy(beta_data)
        self.var2tensor[var] = ops.instance_norm(inp, gamma, beta)

    def _gen_pad(self, var, inps, func):
        inp = self.var2tensor[inps[0]]
        pad = func.body.blocks[0].bindings[0].value

        attrs = pad.attrs
        pads = np.array(attrs.pad_width).reshape([-1, 2]).tolist()
        mode = attrs.pad_mode
        constant_value = float(pad.args[1].data.numpy())
        self.var2tensor[var] = ops.pad(inp, pads, mode, constant_value)

    def _gen_clip(self, var, inps, func):
        inp = self.var2tensor[inps[0]]
        args = func.body.blocks[0].bindings[0].value.args

        self.var2tensor[var] = ops.clip(inp, args[1].value, args[2].value)

    def _gen_concat(self, var, inps, func):
        inp = self.var2tensor[inps[0]]
        axis = int(func.body.blocks[0].bindings[0].value.attrs.axis)
        self.var2tensor[var] = ops.concat(inp, axis)

    def _gen_tuple(self, var, inps):
        inps_list = [self.var2tensor[x] for x in inps]
        self.var2tensor[var] = inps_list
