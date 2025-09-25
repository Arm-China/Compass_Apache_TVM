# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=unused-argument, unsupported-binary-operation
"""Relax IR to Compass IR mapping rules."""
import inspect
from functools import reduce, wraps
from operator import mul
import numpy as np
from tvm import relax, ir, tir, ffi
from tvm.relax.dpl import is_const, is_op, wildcard, is_tuple, is_tuple_get_item
from tvm.relax.transform import PatternCheckContext
from tvm.relax.backend.utils import has_leaking_intermediate_variables
from ..config import CompassConfig


def is_scalar_and_close(x, ref):
    """Check if the given argument is a scalar and close to the given reference value."""
    assert isinstance(x, (relax.Constant, int, float))
    assert isinstance(ref, (int, float))
    if isinstance(x, relax.Constant):
        x = x.data.numpy()
    if x.size != 1:
        return False
    return np.isclose(float(x), float(ref))


def _convert_array(arr):
    if isinstance(arr, (tir.IntImm, int)):
        return int(arr)
    if isinstance(arr, (list, tuple, ffi.Array)):
        return [_convert_array(val) for val in arr]
    return int(arr)


def _check_range(check_list, rule, min_value=1):
    if len(check_list) != len(rule):
        return False
    # ensure that shape should not be dynamic.
    if isinstance(check_list, (list, tuple, ffi.Array)) and _is_dynamic(check_list):
        return False

    return all(min_value <= x <= y for x, y in zip(_convert_array(check_list), rule))


def _check_shape(shape, shape_0_idx, shape_n_idx, largest_dim):
    if _is_dynamic(shape):
        return False

    # The size limit of dim 0
    shape_0_pat = (
        [32],
        [16],
    )
    # The size limit of dim n(dimensions other than the dim 0)
    shape_n_pat = (
        [16384] * 4,
        [1920, 1080, 1920, 4096],
        [100, 1080, 1920, 4096],
    )
    assert -len(shape_0_pat) <= shape_0_idx <= len(shape_0_pat) - 1
    assert -len(shape_n_pat) <= shape_n_idx <= len(shape_n_pat) - 1

    dim = len(shape)
    spec = (
        shape_0_pat[shape_0_idx]
        + shape_n_pat[shape_n_idx][-(largest_dim - 1) :][largest_dim - dim :]
    )
    return _check_range(shape, spec)


def _check_dim(shapes, dims, keep_dims=True):
    shape_dims = set(len(shape) for shape in shapes)
    if not shape_dims.issubset(set(dims)):
        return False
    if keep_dims:
        return len(shape_dims) == 1
    return True


def _check_conv2d(conv2d: relax.Call, add_const: relax.Constant):
    # Check if it is supported by Compass.
    attrs = conv2d.attrs
    in_shape = conv2d.args[0].struct_info.shape.values
    out_shape = conv2d.struct_info.shape.values

    if add_const:
        squeezed_shape = [x for x in add_const.struct_info.shape.values if x != 1]
        if not squeezed_shape:
            squeezed_shape = [1]
        if len(squeezed_shape) != 1 or squeezed_shape[0] != out_shape[-1]:
            return False

    yield  # The Compass OP Spec check code must be placed after this statement.

    res = []
    # only support 4-dim
    res.append(_check_dim([in_shape, out_shape], [4]))
    # check shape of input and output
    res.append(_check_range(in_shape[1:], [16384, 16384, 16384]))
    res.append(_check_range(out_shape[1:], [16384, 16384, 16384]))
    # check kernel
    k_layout = attrs.kernel_layout
    k_shape = conv2d.args[1].struct_info.shape.values
    kernels = [ks for layout, ks in zip(k_layout, k_shape) if layout in ("H", "W")]
    res.append(_check_range(kernels, [64, 64]))
    # check stride
    strides = attrs.strides
    res.append(_check_range(strides, [16, 16]))
    # check pad
    # can actually support larger pads, just skip
    # pads = attrs.padding
    # res.append(_check_range(pads, [16, 16, 16, 16], 0))
    # check dilation
    dilation = attrs.dilation
    res.append(_check_range(dilation, [40, 40]))
    if dilation[0] != 1 or dilation[1] != 1:
        # Here need to multiply sizeof(output_type), but cannnot get quantified type.
        # So use 16bit as default.
        res.append(reduce(mul, out_shape) * reduce(mul, strides) * 2 <= 1024 * 1024 * 1024)

    return all(res)


def _is_dynamic(shape_or_type):
    # TODO: Refactor.
    # def _get_shapes(ir_type):
    #     if isinstance(ir_type, ir.TensorType):
    #         return [ir_type.shape]
    #     if isinstance(ir_type, ir.TupleType):
    #         shapes = []
    #         for field in ir_type.fields:
    #             shapes += _get_shapes(field)
    #         return shapes
    #     return []

    # shapes = _get_shapes(shape_or_type) if isinstance(shape_or_type, ir.Type) else [shape_or_type]
    # for shape in shapes:
    #     if any(isinstance(dim, tir.Var) for dim in shape):
    #         return True
    return False


def _dq(x):
    return is_op("relax.dequantize")(x, is_const(), is_const())


def _quant(x):
    return is_op("relax.quantize")(x, is_const(), is_const())


def _checker(func):
    func_name = func.__name__

    @wraps(func)
    def _wrapper(context: PatternCheckContext):
        call = context.annotated_expr["root"]
        if isinstance(call, relax.Tuple):
            in_types = []
            for ele in call:
                in_types += [x.checked_type for x in ele.args]
        else:
            in_types = [x.checked_type for x in call.args]
        types = in_types + [call.checked_type]
        if any(_is_dynamic(x) for x in types):
            return False

        ret = func(context)
        is_generator = inspect.isgenerator(ret)

        if is_generator:
            generator = ret
            try:
                # Execute the code before the "yield" statement.
                next(generator)
            except StopIteration as e:  # pylint: disable=invalid-name
                value = e.value
                assert isinstance(value, bool), f'The return type of "{func_name}" must be "bool".'
                return value

        # The code used to check the Compass OP Spec can be skipped by users,
        # for the generator function, they are the code after the "yield"
        # statement, otherwise, they are the whole function body.
        if CompassConfig.get().common["disable_op_spec_checker"] == "true":
            return True

        if not is_generator:
            return ret

        try:
            # Execute the code after the "yield" statement.
            next(generator)
        except StopIteration as e:  # pylint: disable=invalid-name
            assert isinstance(e.value, bool), f'The return type of "{func_name}" must be "bool".'
            return e.value
        raise RuntimeError(f'There is more than one "yield" in "{func_name}".')

    return _wrapper


def _convolution2d_pattern():
    conv = is_op("relax.nn.conv2d")(wildcard(), is_const()) | is_op("relax.nn.conv2d_transpose")(
        wildcard(), is_const()
    )
    add_const = is_const()
    conv_add = is_op("relax.add")(conv, add_const)
    conv_add_relu = is_op("relax.nn.relu")(conv_add)
    conv_add_leaky_relu = is_op("relax.nn.leakyrelu")(conv_add)
    conv_add_clip = is_op("relax.clip")(conv_add, wildcard(), wildcard())
    conv_leaky_relu = is_op("relax.nn.leakyrelu")(conv)
    conv_relu = is_op("relax.nn.relu")(conv)

    pattern = (
        conv
        | conv_add
        | conv_add_relu
        | conv_add_leaky_relu
        | conv_add_clip
        | conv_leaky_relu
        | conv_relu
    )

    annotations = {
        "root": pattern,
        "conv": conv,
        "add_const": add_const,
        "clip": conv_add_clip,
    }

    @_checker
    def _check(ctx: PatternCheckContext) -> bool:
        if has_leaking_intermediate_variables(ctx):
            return False
        # Retrieve the annotated expression from context
        clip = ctx.annotated_expr.get("clip")
        if clip:
            args = clip.args
            is_min_0 = args[1].value.value == 0.0
            is_max_6 = args[2].value.value == 6.0
            if not (is_min_0 and is_max_6):
                return False

        conv2d = ctx.annotated_expr.get("conv")
        add_const = ctx.annotated_expr.get("add_const")

        return _check_conv2d(conv2d, add_const)

    return ("compass.conv2d", pattern, annotations, _check)


def _qnn_conv2d_pattern():
    conv = is_op("relax.nn.conv2d")(_dq(wildcard()), _dq(is_const()))
    add_const = is_const()
    conv_add = is_op("relax.add")(conv, _dq(add_const))
    q_conv_add = _quant(conv_add)
    q_conv_add_qleakyrelu = _quant(is_op("relax.nn.leakyrelu")(_dq(q_conv_add)))
    conv_add_relu = is_op("relax.nn.relu")(conv_add)
    conv_add_clip = is_op("relax.clip")(conv_add, wildcard(), wildcard())

    convt = is_op("relax.nn.conv2d_transpose")(_dq(wildcard()), _dq(is_const()))
    convt_add = is_op("relax.add")(convt, _dq(add_const))
    qconvt = _quant(convt)
    qconvt_qleakyrelu = _quant(is_op("relax.nn.leakyrelu")(_dq(qconvt)))

    pattern = (
        q_conv_add
        | _quant(conv_add_relu)
        | _quant(conv_add_clip)
        | q_conv_add_qleakyrelu
        | _quant(convt_add)
        | qconvt_qleakyrelu
    )

    annotations = {
        "root": pattern,
        "conv": conv,
        "convt": convt,
        "add_const": add_const,
        "clip": conv_add_clip,
    }

    @_checker
    def _check(ctx: PatternCheckContext) -> bool:
        if has_leaking_intermediate_variables(ctx):
            return False
        clip = ctx.annotated_expr.get("clip")
        if clip:
            args = clip.args
            is_min_0 = args[1].value.value == 0.0
            is_max_6 = args[2].value.value == 6.0
            if not (is_min_0 and is_max_6):
                return False
        conv2d = ctx.annotated_expr.get("conv") or ctx.annotated_expr.get("convt")
        add_const = ctx.annotated_expr.get("add_const")
        return _check_conv2d(conv2d, add_const)

    return ("compass.qnn.conv2d", pattern, annotations, _check)


def _instance_norm_pattern():
    inp = wildcard()
    mean = is_op("relax.mean")(inp)
    sub = is_op("relax.subtract")(inp, mean)
    var = is_op("relax.variance")(inp)
    add = is_op("relax.add")(var, is_const())
    sqrt = is_op("relax.sqrt")(add)
    divide = is_op("relax.divide")(sub, sqrt)
    multiply = is_op("relax.multiply")(divide, is_const())
    out = is_op("relax.add")(multiply, is_const())
    annotations = {"root": out, "input": inp}

    @_checker
    def _check(ctx: PatternCheckContext) -> bool:
        in_shape = ctx.annotated_expr["input"].struct_info.shape.values
        out_shape = ctx.annotated_expr["root"].struct_info.shape.values

        res = []
        # only support 4-dim
        res.append(_check_dim([in_shape, out_shape], [4]))
        # check shape of input and output
        res.append(_check_range(in_shape[1:], [16384, 16384, 16384]))
        res.append(_check_range(out_shape[1:], [16384, 16384, 16384]))
        return all(res)

    return ("compass.instance_norm", out, annotations, _check)


def _layer_norm_pattern0():
    inp = wildcard()
    mean0 = is_op("relax.mean")(inp)
    sub = is_op("relax.subtract")(inp, mean0)
    power = is_op("relax.power")(sub, is_const())
    mean1 = is_op("relax.mean")(power)
    add = is_op("relax.add")(mean1, is_const())
    sqrt = is_op("relax.sqrt")(add)
    divide = is_op("relax.divide")(sub, sqrt)
    multiply = is_op("relax.multiply")(divide, is_const())
    out = is_op("relax.add")(multiply, is_const())
    annotations = {
        "root": out,
        "input": inp,
        "mean0": mean0,
        "mean1": mean1,
        "power": power,
        "add": add,
    }

    @_checker
    def _check(ctx: PatternCheckContext) -> bool:
        mean0 = ctx.annotated_expr["mean0"]
        mean1 = ctx.annotated_expr["mean1"]
        power = ctx.annotated_expr["power"]
        add = ctx.annotated_expr["add"]

        if not ir.structural_equal(mean0.attrs.axis, mean1.attrs.axis):
            return False
        if power.args[1].data.numpy() != 2.0:
            return False
        if add.args[1].data.numpy() > 1e-4:
            return False
        in_shape = ctx.annotated_expr["input"].struct_info.shape.values
        dim = len(in_shape)
        axis = int(mean0.attrs.axis[0])
        if len(mean0.attrs.axis) != 1 or (axis != -1 and axis != dim - 1):
            return False

        yield

        res = []
        res.append(_check_dim([in_shape], [2, 3, 4, 5, 6]))
        return all(res)

    return ("compass.layer_norm0", out, annotations, _check)


def _layer_norm_pattern1():
    inp = wildcard()
    mean0 = is_op("relax.mean")(inp)
    sub = is_op("relax.subtract")(inp, mean0)

    var = is_op("relax.variance")(inp)
    add = is_op("relax.add")(var, is_const())
    sqrt = is_op("relax.sqrt")(add)

    divide = is_op("relax.divide")(sub, sqrt)
    multiply = is_op("relax.multiply")(divide, is_const())
    out = is_op("relax.add")(multiply, is_const())
    annotations = {
        "root": out,
        "input": inp,
        "mean0": mean0,
        "var": var,
        "multiply": multiply,
        "add": add,
    }

    @_checker
    def _check(ctx: PatternCheckContext) -> bool:
        mean0 = ctx.annotated_expr["mean0"]
        var = ctx.annotated_expr["var"]
        add = ctx.annotated_expr["add"]

        if not ir.structural_equal(mean0.attrs.axis, var.attrs.axis):
            return False
        if add.args[1].data.numpy() > 1e-4:
            return False
        in_shape = ctx.annotated_expr["input"].struct_info.shape.values
        dim = len(in_shape)
        axis = int(mean0.attrs.axis[0])
        if len(mean0.attrs.axis) != 1 or (axis != -1 and axis != dim - 1):
            return False

        yield

        res = []
        res.append(_check_dim([in_shape], [2, 3, 4, 5, 6]))
        return all(res)

    return ("compass.layer_norm1", out, annotations, _check)


def _batch_norm_pattern():
    inp = wildcard()
    sub_const, div_const, mul_const, add_const = is_const(), is_const(), is_const(), is_const()
    sub = is_op("relax.subtract")(inp, sub_const)
    divide = is_op("relax.divide")(sub, div_const)
    multiply = is_op("relax.multiply")(divide, mul_const)
    out = is_op("relax.add")(multiply, add_const)
    annotations = {
        "root": out,
        "sub_const": sub_const,
        "div_const": div_const,
        "mul_const": mul_const,
        "add_const": add_const,
    }

    @_checker
    def _check(ctx: PatternCheckContext) -> bool:
        sub_const = ctx.annotated_expr["sub_const"]
        div_const = ctx.annotated_expr["div_const"]
        mul_const = ctx.annotated_expr["mul_const"]
        add_const = ctx.annotated_expr["add_const"]

        for const_in in [sub_const, div_const, mul_const, add_const]:
            shape = [int(val) for val in const_in.struct_info.shape]
            if len(shape) != 1 and not all([val == 1 for val in shape[:-1]]):
                return False

        yield

        return True

    return ("compass.batch_norm", out, annotations, _check)


@_checker
def _check_rms_norm(ctx: PatternCheckContext) -> bool:
    rms_norm = ctx.annotated_expr["root"]
    in_shape = rms_norm.args[0].struct_info.shape.values

    res = []
    # support 2-6 dim
    res.append(_check_dim([in_shape], [2, 3, 4, 5, 6]))
    return all(res)


@_checker
def _check_hard_swish(ctx: PatternCheckContext) -> bool:
    clip = ctx.annotated_expr["clip"]
    add_const = ctx.annotated_expr["add_const"]
    mul_const = ctx.annotated_expr.get("mul_const")
    div_const = ctx.annotated_expr.get("div_const")

    is_min_0 = clip.args[1].value.value == 0.0
    is_max_6 = clip.args[2].value.value == 6.0
    if not (is_min_0 and is_max_6):
        return False
    if not is_scalar_and_close(add_const, 3):
        return False
    if mul_const and not is_scalar_and_close(mul_const, 1 / 6):
        return False
    if div_const and not is_scalar_and_close(div_const, 6):
        return False

    yield

    data = ctx.annotated_expr["inp"]
    in_shape = data.struct_info.shape.values
    res = []
    res.append(_check_dim([in_shape], [2, 3, 4]))
    res.append(_check_shape(in_shape, 0, 0, 4))
    return all(res)


def _hard_swish_pattern():
    inp = wildcard()
    add_const = is_const()
    add = is_op("relax.add")(inp, add_const)
    add_clip = is_op("relax.clip")(add, wildcard(), wildcard())
    add_clip_mul = is_op("relax.multiply")(inp, add_clip)
    mul_const, div_const = is_const(), is_const()
    add_clip_mul_mul = is_op("relax.multiply")(add_clip_mul, mul_const)
    add_clip_mul_div = is_op("relax.divide")(add_clip_mul, div_const)
    pattern = add_clip_mul_mul | add_clip_mul_div
    annotations = {
        "root": pattern,
        "inp": inp,
        "clip": add_clip,
        "div_const": div_const,
        "mul_const": mul_const,
        "add_const": add_const,
    }

    return ("compass.hard_swish", pattern, annotations, _check_hard_swish)


def _qnn_hard_swish_pattern():
    inp = wildcard()
    add_const = is_const()
    add = is_op("relax.add")(_dq(inp), add_const)
    add_clip = is_op("relax.clip")(add, wildcard(), wildcard())
    add_clip_mul = is_op("relax.multiply")(_dq(inp), add_clip)
    mul_const, div_const = is_const(), is_const()
    add_clip_mul_mul = is_op("relax.multiply")(add_clip_mul, mul_const)
    add_clip_mul_div = is_op("relax.divide")(add_clip_mul, div_const)
    pattern = _quant(add_clip_mul_mul) | _quant(add_clip_mul_div)
    annotations = {
        "root": pattern,
        "inp": inp,
        "clip": add_clip,
        "div_const": div_const,
        "mul_const": mul_const,
        "add_const": add_const,
    }

    return ("compass.qnn.hard_swish", pattern, annotations, _check_hard_swish)


def _batch_norm_single_pattern():
    arg_in = wildcard()
    const_in = is_const()
    add = is_op("relax.add")(arg_in, const_in)
    multiply = is_op("relax.multiply")(arg_in, const_in)
    sub = is_op("relax.subtract")(arg_in, const_in)
    div = is_op("relax.divide")(arg_in, const_in)
    out = add | multiply | sub | div
    annotations = {"root": out, "arg_in": arg_in, "const_in": const_in}

    @_checker
    def _check(ctx: PatternCheckContext) -> bool:
        call = ctx.annotated_expr["root"]
        arg_in = ctx.annotated_expr["arg_in"]
        const_in = ctx.annotated_expr["const_in"]
        if isinstance(arg_in, relax.Constant):
            return False
        if len(arg_in.struct_info.shape) not in (3, 4):
            return False
        name = call.op.name[6:]
        if name in ["add", "subtract"] and np.allclose(const_in.data.numpy(), 0):
            return False
        if name in ["multiply", "divide"] and np.allclose(const_in.data.numpy(), 1):
            return False

        shape = [int(val) for val in const_in.struct_info.shape]
        if len(shape) != 1 and not all([val == 1 for val in shape[:-1]]):
            return False

        yield
        return True

    return ("compass.batch_norm_single", out, annotations, _check)


def _dense_pattern():
    inp = wildcard()
    out = is_op("relax.matmul")(inp, is_const())
    annotations = {"root": out, "inp": inp}

    @_checker
    def _check(ctx: PatternCheckContext) -> bool:
        return _check_dim([ctx.annotated_expr["inp"].struct_info.shape], [2])

    return ("compass.dense", out, annotations, _check)


@_checker
def _check_matmul(ctx: PatternCheckContext) -> bool:
    """Batch matmul"""
    if has_leaking_intermediate_variables(ctx):
        return False

    matmul = ctx.annotated_expr["root"]
    in_shape1 = [int(val) for val in matmul.args[0].struct_info.shape.values]
    in_shape2 = [int(val) for val in matmul.args[1].struct_info.shape.values]

    res = []
    # check shape of input and output
    res.append(_check_range(in_shape1[-2:], [16384] * 2))
    res.append(_check_range(in_shape2[-2:], [16384] * 2))
    return all(res)


@_checker
def _check_pad(ctx: PatternCheckContext) -> bool:
    # Check if it is supported by Compass.
    pad = ctx.annotated_expr["root"]
    # Check pad_mod.
    attrs = pad.attrs
    if attrs.pad_mode.lower() not in ["constant", "reflect", "symmetric"]:
        return False

    yield  # The Compass OP Spec check code must be placed after this statement.

    in_shape = pad.args[0].struct_info.shape.values
    out_shape = pad.struct_info.shape.values
    res = []
    # support 2, 3, 4, 5 dim
    dim = len(in_shape)
    res.append(_check_dim([in_shape, out_shape], [2, 3, 4, 5]))
    res.append(_check_range(in_shape[1:], [16384] * (dim - 1)))

    # check pad_width
    pad_width = attrs.pad_width
    res.append(_check_range(pad_width, [128] * len(pad_width), -1))
    return all(res)


def _elementwise_relu_pattern():
    pattern_op = is_op("relax.add") | is_op("relax.subtract") | is_op("relax.multiply")
    pattern = pattern_op(wildcard(), wildcard())
    pattern = is_op("relax.nn.relu")(pattern)

    annotations = {"root": pattern}

    @_checker
    def _check(ctx: PatternCheckContext) -> bool:
        call = ctx.annotated_expr["root"]
        # This pattern will be codegen to 1 Compass OP, so we don't want it to have multiple usages.
        if len(ctx.var_usages.get(call.args[0], [])) > 1:
            return False
        return True

    return ("compass.eltwise_relu", pattern, annotations, _check)


def _qnn_elementwise_relu_pattern():
    eltwise_op = is_op("relax.add") | is_op("relax.subtract") | is_op("relax.multiply")
    eltwise_op |= is_op("relax.minimum")
    eltwise = _quant(eltwise_op(_dq(wildcard()), _dq(wildcard())))
    relu = is_op("relax.nn.relu")(_dq(eltwise))
    pattern = _quant(relu)
    annotations = {"root": pattern}

    return ("compass.qnn.eltwise_relu", pattern, annotations)


def _qnn_silu_pattern():
    inp = wildcard()
    sigmoid = _quant(is_op("relax.sigmoid")(_dq(inp)))
    multiply = is_op("relax.multiply")(_dq(inp), _dq(sigmoid))
    pattern = _quant(multiply)
    annotations = {"root": pattern}

    return ("compass.qnn.silu", pattern, annotations)


def _check_matmul_add(inp, bias):
    res = []
    bias_shape = bias.struct_info.shape
    res.append(len(bias_shape) == 1 or int(bias_shape[0]) == 1)
    in_shape = inp.struct_info.shape
    res.append(_check_dim([in_shape], [2]))
    return all(res)


def _matmul_add_pattern():
    inp = wildcard()
    matmul = is_op("relax.matmul")(inp, is_const())
    add_const = is_const()
    add = is_op("relax.add")(matmul, add_const)
    relu = is_op("relax.nn.relu")(add)
    pattern = add | relu
    annotations = {"root": pattern, "add_const": add_const, "inp": inp}

    @_checker
    def _check(ctx: PatternCheckContext) -> bool:
        # Check if the given match is supported by Compass.
        inp = ctx.annotated_expr["inp"]
        add_const = ctx.annotated_expr["add_const"]
        return _check_matmul_add(inp, add_const)

    return ("compass.matmul_add", pattern, annotations, _check)


def _qnn_matmul_add_pattern():
    inp = wildcard()
    matmul = is_op("relax.matmul")(_dq(inp), _dq(is_const()))
    add_const = is_const()
    add = is_op("relax.add")(matmul, _dq(add_const))
    pattern = _quant(add)
    annotations = {"root": pattern, "inp": inp, "add_const": add_const}

    @_checker
    def _check(ctx: PatternCheckContext) -> bool:
        # Check if the given match is supported by Compass.
        inp = ctx.annotated_expr["inp"]
        add_const = ctx.annotated_expr["add_const"]
        return _check_matmul_add(inp, add_const)

    return ("compass.qnn.matmul_add", pattern, annotations, _check)


def _qnn_squared_difference_pattern():
    sub = is_op("relax.subtract")(_dq(wildcard()), _dq(wildcard()))
    multiply = is_op("relax.multiply")(sub, sub)
    pattern = _quant(multiply)
    annotations = {"root": pattern}

    return ("compass.qnn.squared_diff", pattern, annotations)


def _qnn_basic_lstm_pattern():
    inp0, inp1, inp2 = wildcard(), wildcard(), wildcard()
    q_matmul0 = _quant(is_op("relax.matmul")(_dq(inp0), _dq(is_const())))
    q_matmul1 = _quant(is_op("relax.matmul")(_dq(inp1), _dq(is_const())))
    q_add0 = _quant(is_op("relax.add")(_dq(q_matmul0), _dq(q_matmul1)))
    q_add1 = _quant(is_op("relax.add")(_dq(q_add0), _dq(is_const())))
    split = is_op("relax.split")(q_add1)
    split0 = is_tuple_get_item(split, 0)
    split1 = is_tuple_get_item(split, 1)
    split2 = is_tuple_get_item(split, 2)
    split3 = is_tuple_get_item(split, 3)
    q_sigmoid0 = _quant(is_op("relax.sigmoid")(_dq(split0)))
    q_sigmoid1 = _quant(is_op("relax.sigmoid")(_dq(split1)))
    q_sigmoid2 = _quant(is_op("relax.sigmoid")(_dq(split3)))
    q_tanh0 = _quant(is_op("relax.tanh")(_dq(split2)))
    q_mul0 = _quant(is_op("relax.multiply")(_dq(q_sigmoid1), _dq(inp2)))
    q_mul1 = _quant(is_op("relax.multiply")(_dq(q_sigmoid0), _dq(q_tanh0)))
    q_add2 = _quant(is_op("relax.add")(_dq(q_mul0), _dq(q_mul1)))
    q_tanh1 = _quant(is_op("relax.tanh")(_dq(q_add2)))
    q_mul3 = _quant(is_op("relax.multiply")(_dq(q_sigmoid2), _dq(q_tanh1)))

    pattern = q_mul3
    annotations = {
        "root": pattern,
        "inp0": inp0,
        "inp1": inp1,
        "inp2": inp2,
        "out0": q_add2,
    }

    @_checker
    def _check(ctx: PatternCheckContext) -> bool:
        # Check if the given match is supported by Compass.
        inp0 = ctx.annotated_expr["inp0"]
        inp1 = ctx.annotated_expr["inp1"]
        inp2 = ctx.annotated_expr["inp2"]
        out0 = ctx.annotated_expr["out0"]
        out1 = ctx.annotated_expr["root"]
        in_shape0 = inp0.struct_info.shape.values
        in_shape1 = inp1.struct_info.shape.values
        in_shape2 = inp2.struct_info.shape.values
        out_shape0 = out0.struct_info.shape.values
        out_shape1 = out1.struct_info.shape.values

        res = []
        # only support 2-dim(in_shape0 will be reshaped to 3 dims in the codegen)
        res.append(_check_dim([in_shape0, in_shape1, in_shape2, out_shape0, out_shape1], [2]))
        # check shape of input
        res.append(_check_range(in_shape0[:], [3071, 3072]))
        res.append(_check_range(in_shape1[:], [32, 3072]))
        res.append(_check_range(in_shape2[:], [32, 3072]))

        return all(res)

    return ("compass.qnn.basic_lstm", pattern, annotations, _check)


def _softplus_pattern():
    exp = is_op("relax.exp")(wildcard())
    add_const = is_const()
    add = is_op("relax.add")(exp, add_const)
    pattern = is_op("relax.log")(add)
    annotations = {"root": pattern, "add_const": add_const}

    @_checker
    def _check(ctx: PatternCheckContext) -> bool:
        # Check if the given match is supported by Compass.
        add_const = ctx.annotated_expr["add_const"]
        if not is_scalar_and_close(add_const, 1):
            return False
        yield  # The Compass op Spec check code must be placed after this statement.
        return True

    return ("compass.softplus", pattern, annotations, _check)


@_checker
def _check_transpose(ctx: PatternCheckContext) -> bool:
    transpose = ctx.annotated_expr["root"]
    in_shape = transpose.args[0].struct_info.shape.values
    res = []
    # support 1, 2, 3, 4, 5, 6 dim
    res.append(_check_dim([in_shape], [1, 2, 3, 4, 5, 6]))
    return all(res)


@_checker
def _check_max_pool2d(ctx: PatternCheckContext) -> bool:
    max_pool2d = ctx.annotated_expr["root"]
    in_shape = max_pool2d.args[0].struct_info.shape.values
    attrs = max_pool2d.attrs

    res = []
    # only support 4-dim
    res.append(_check_dim([in_shape], [4]))
    # check shape of input
    res.append(_check_range(in_shape[:], [32, 16384, 1920, 16384]))
    # check kernel
    kernels = attrs.pool_size
    res.append(_check_range(kernels, [65, 65]))
    # check stride
    strides = attrs.strides
    res.append(_check_range(strides, [16, 16]))
    res.append(
        (strides[0] <= kernels[0] and strides[1] <= kernels[1])
        or (kernels[0] == kernels[1] == 1 and strides[0] == strides[1] == 2)
    )
    # check pad
    pads = attrs.padding
    res.append(_check_range(pads, [16, 16, 16, 16], 0))
    res.append(all(p < kernels[0] for p in pads[::2]) and all(p < kernels[1] for p in pads[1::2]))

    return all(res)


@_checker
def _check_avg_pool2d(ctx: PatternCheckContext) -> bool:
    avg_pool2d = ctx.annotated_expr["root"]
    in_shape = avg_pool2d.args[0].struct_info.shape.values
    attrs = avg_pool2d.attrs

    res = []
    # only support 4-dim
    res.append(_check_dim([in_shape], [4]))
    # check shape of input
    res.append(_check_range(in_shape[:], [32, 16384, 1920, 16384]))
    # check kernel
    kernels = attrs.pool_size
    max_int32 = 2**31 - 1
    input_data_size = 32
    res.append(
        _check_range(kernels, [65, 65])
        or (
            kernels[1] == in_shape[2]
            and kernels[0] == in_shape[1]
            and (
                kernels[1] * kernels[0] < 28 * 1024
                or kernels[1] * kernels[0] * input_data_size < max_int32
            )
        )
    )
    # check stride
    strides = attrs.strides
    res.append(
        _check_range(strides, [32, 32]) or (kernels[1] == in_shape[2] and kernels[0] == in_shape[1])
    )
    res.append(
        (strides[0] <= kernels[0] and strides[1] <= kernels[1])
        or (kernels[0] == kernels[1] == 1 and strides[0] == strides[1] == 2)
    )
    # check pad
    pads = attrs.padding
    res.append(_check_range(pads, [6, 6, 6, 6], 0))
    res.append(all(p < kernels[0] for p in pads[::2]) and all(p < kernels[1] for p in pads[1::2]))

    return all(res)


@_checker
def _check_sign(ctx: PatternCheckContext) -> bool:
    call = ctx.annotated_expr["root"]
    in_shape0 = call.args[0].struct_info.shape.values

    res = []
    res.append(_check_dim([in_shape0], [2, 3, 4]))
    res.append(_check_shape(in_shape0, 0, 0, 4))
    return all(res)


@_checker
def _check_concat(ctx: PatternCheckContext) -> bool:
    call = ctx.annotated_expr["root"]
    res = []
    res.append(len(call.args[0]) >= 2)
    return all(res)


@_checker
def _check_space_to_depth(ctx: PatternCheckContext) -> bool:
    call = ctx.annotated_expr["root"]
    in_shape0 = call.args[0].struct_info.shape.values
    if len(in_shape0) != 4:
        return False
    yield
    res = []
    res.append(_check_shape(in_shape0, 0, 0, 4))
    res.append(1 <= call.attrs.block_size <= 16)
    return all(res)


def _gen_qnn_image_resize2d():
    inp = wildcard()
    resize2d = is_op("relax.image.resize2d")(_dq(inp), wildcard())
    out = _quant(resize2d)
    annotations = {"root": out, "inp": inp, "call": resize2d}

    @_checker
    def _check(ctx: PatternCheckContext) -> bool:
        resize2d = ctx.annotated_expr["call"]
        inp = ctx.annotated_expr["inp"]
        in_shape = inp.struct_info.shape.values
        attrs = resize2d.attrs

        res = []
        # only support 4-dim
        res.append(_check_dim([in_shape], [4]))
        # check shape of input
        res.append(_check_range(in_shape[1:], [1080, 1920, 16384]))
        # check method
        res.append(attrs.method in ["nearest_neighbor", "linear"])
        # check mode
        res.append(
            attrs.coordinate_transformation_mode
            in [
                "half_pixel",
                "align_corners",
                "asymmetric",
                "pytorch_half_pixel",
                "tf_half_pixel_for_nn",
            ]
        )

        return all(res)

    return ("compass.qnn.image.resize2d", out, annotations, _check)


def _gen_qnn_concat(candidate_inp_num):
    qnn_concat_patterns = []
    for num in candidate_inp_num:
        inp = tuple(_dq(wildcard()) for _ in range(num))
        concat = is_op("relax.concat")(is_tuple(inp))
        out = _quant(concat)
        annotations = {"root": out}
        pattern = ("compass.qnn.concat", out, annotations)
        qnn_concat_patterns.append(pattern)

    return qnn_concat_patterns


@_checker
def _check_depth_to_space(ctx: PatternCheckContext) -> bool:
    call = ctx.annotated_expr["root"]
    in_shape0 = call.args[0].struct_info.shape.values
    if len(in_shape0) != 4:
        return False
    yield
    res = []
    res.append(_check_shape(in_shape0, 0, 1, 4))
    res.append(1 <= call.attrs.block_size <= 16)

    return all(res)


@_checker
def _check_resize2d(ctx: PatternCheckContext) -> bool:
    resize2d = ctx.annotated_expr["root"]
    in_shape = resize2d.args[0].struct_info.shape.values
    attrs = resize2d.attrs

    res = []
    # only support 4-dim
    res.append(_check_dim([in_shape], [4]))
    # check shape of input
    res.append(_check_range(in_shape[1:], [1080, 1920, 16384]))
    # check method
    res.append(attrs.method in ["nearest_neighbor", "linear"])
    # check mode
    res.append(
        attrs.coordinate_transformation_mode
        in [
            "half_pixel",
            "align_corners",
            "asymmetric",
            "pytorch_half_pixel",
            "tf_half_pixel_for_nn",
        ]
    )

    return all(res)


@_checker
def _check_softmax(ctx: PatternCheckContext) -> bool:
    softmax = ctx.annotated_expr["root"]
    in_sinfo = softmax.args[0].struct_info

    res = []
    res.append(_check_dim([in_sinfo.shape.values], [1, 2, 3, 4, 5, 6]))
    res.append(-1 <= softmax.attrs.axis <= in_sinfo.ndim - 1)
    return all(res)


@_checker
def _check_scatter_nd(ctx: PatternCheckContext) -> bool:
    scatter_nd = ctx.annotated_expr["root"]
    if not isinstance(scatter_nd.args[1], relax.Constant):
        return False

    yield

    i_shape0 = [int(x) for x in scatter_nd.args[0].struct_info.shape]
    i_shape1 = [int(x) for x in scatter_nd.args[1].struct_info.shape]
    i_shape2 = [int(x) for x in scatter_nd.args[2].struct_info.shape]
    res = []
    res.append(_check_dim([i_shape0], [1, 2, 3, 4, 5]))
    res.append(i_shape1[-1] <= len(i_shape0))
    res.append(all(i == j for i, j in zip(i_shape2, i_shape1[:-1] + i_shape0[i_shape1[-1] :])))
    return all(res)


@_checker
def _check_erf(ctx: PatternCheckContext) -> bool:
    erf = ctx.annotated_expr["root"]
    in_shape = erf.args[0].struct_info.shape.values

    res = []
    # support 2, 3, 4 dim
    res.append(_check_dim([in_shape], [2, 3, 4]))
    # check shape of input
    res.append(_check_shape(in_shape, 0, 0, 4))
    return all(res)


@_checker
def _check_split(ctx: PatternCheckContext) -> bool:
    split = ctx.annotated_expr["root"]
    in_shape = split.args[0].struct_info.shape.values
    out_shapes = [tt.shape.values for tt in split.struct_info.fields]

    res = []
    # check output tensor number
    res.append(len(out_shapes) >= 2)
    # support 2, 3, 4, 5 dim
    res.append(_check_dim([in_shape] + out_shapes, [2, 3, 4, 5]))
    dim = len(in_shape)
    res.append(_check_range(in_shape[:], [128] + [16384] * (dim - 1)))
    return all(res)


@_checker
def _check_s2b_nd(ctx: PatternCheckContext) -> bool:
    s2b_nd = ctx.annotated_expr["root"]
    in_shape = s2b_nd.args[0].struct_info.shape.values
    if len(in_shape) != 4:
        return False

    yield

    res = []
    res.append(_check_shape(in_shape, 0, 0, 4))
    for block_size in s2b_nd.attrs.block_shape:
        res.append(1 <= int(block_size) <= 16)
    for pad in s2b_nd.attrs.paddings:
        for size in pad:
            res.append(0 <= int(size) <= 16)
    # Ops api unsupport set pad value for s2b.
    res.append(s2b_nd.attrs.pad_value == 0.0)

    return all(res)


@_checker
def _check_b2s_nd(ctx: PatternCheckContext) -> bool:
    b2s_nd = ctx.annotated_expr["root"]
    in_shape = b2s_nd.args[0].struct_info.shape.values
    if len(in_shape) != 4:
        return False

    yield

    res = []
    res.append(_check_shape(in_shape, 0, 0, 4))
    for block_size in b2s_nd.attrs.block_shape:
        res.append(1 <= int(block_size) <= 16)
    for crop in b2s_nd.attrs.crops:
        for size in crop:
            res.append(0 <= int(size) <= 16)

    return all(res)


@_checker
def _check_flip(ctx: PatternCheckContext) -> bool:
    flip = ctx.annotated_expr["root"]
    in_shape = flip.args[0].struct_info.shape.values

    yield

    res = []
    # support 2, 3 dim
    res.append(_check_dim([in_shape], [2, 3]))
    res.append(_check_shape(in_shape, 0, 0, 3))
    return all(res)


@_checker
def _check_ctc_greedy_decoder(ctx: PatternCheckContext) -> bool:
    ctc = ctx.annotated_expr["root"]
    data_shape = ctc.args[0].struct_info.shape.values
    seq_shape = ctc.args[1].struct_info.shape.values
    out_shape = ctc.struct_info.shape.values

    res = []
    # check dim
    res.append(_check_dim([data_shape], [3]))
    res.append(_check_dim([seq_shape], [1]))
    res.append(_check_dim([out_shape], [4]))
    # check shape of input and output
    res.append(_check_range(data_shape, [32, 16384, 16384]))
    res.append(_check_range(seq_shape, [32]))
    res.append(1 <= out_shape[0] <= 32)
    res.append(out_shape[1] == 4096 and out_shape[2] == out_shape[3] == 1)
    res.append(bool(ctc.attrs.merge_repeated))
    return all(res)


@_checker
def _check_requantize(ctx: PatternCheckContext) -> bool:
    call = ctx.annotated_expr["root"]
    attrs = call.attrs
    dim = len(call.args[0].struct_info.shape)
    if attrs.axis not in [-1, dim - 1]:
        return False

    yield

    res = []
    res.append(attrs.out_dtype in ["int8", "uint8", "int16", "uint16"])

    return all(res)


@_checker
def _check_lrn(ctx: PatternCheckContext) -> bool:
    call = ctx.annotated_expr["root"]
    attrs = call.attrs
    in_shape = call.args[0].struct_info.shape.values
    out_shape = call.struct_info.shape.values

    res = []
    # only support 4-dim
    res.append(_check_dim([in_shape, out_shape], [4]))
    # check shape of input and output
    res.append(_check_shape(in_shape, 0, 0, 4))
    # check size
    res.append(1 <= attrs.size <= 64)
    # check bias
    res.append(0 < attrs.bias <= 64)
    # check alpha
    res.append(0 < attrs.alpha <= 64)
    # check beta
    res.append(0 < attrs.beta < 1)
    # check axis which not support N
    res.append(attrs.axis != 0)
    return all(res)


@_checker
def _check_add(ctx: PatternCheckContext) -> bool:
    call = ctx.annotated_expr["root"]
    if isinstance(call.args[1], relax.Constant):
        return not np.allclose(call.args[1].data.numpy(), 0)
    return True


@_checker
def _check_mul(ctx: PatternCheckContext) -> bool:
    call = ctx.annotated_expr["root"]
    if isinstance(call.args[1], relax.Constant):
        return not np.allclose(call.args[1].data.numpy(), 1)
    return True


@_checker
def _check_prelu(ctx: PatternCheckContext) -> bool:
    call = ctx.annotated_expr["root"]
    in_dim = call.args[0].struct_info.ndim
    if call.attrs.axis not in (in_dim - 1, -1):
        return False
    return True


@_checker
def _check_channel_shuffle(ctx: PatternCheckContext) -> bool:
    call = ctx.annotated_expr["root"]
    in_shape = [int(x) for x in call.args[0].struct_info.shape]
    out_shape = [int(x) for x in call.struct_info.shape]

    res = []
    # support 2, 3, 4 dim
    res.append(_check_dim([in_shape, out_shape], [2, 3, 4]))
    # check shape of input and output
    res.append(_check_shape(in_shape, 0, 0, 4))
    res.append(_check_shape(out_shape, 0, 0, 4))
    # check group
    res.append(1 <= call.attrs.group <= 16384)
    return all(res)


@_checker
def _check_one_hot(ctx: PatternCheckContext) -> bool:
    call = ctx.annotated_expr["root"]
    in_shape = [int(x) for x in call.args[0].struct_info.shape]
    on_value = call.args[1]
    off_value = call.args[2]
    attrs = call.attrs
    dim = len(in_shape)

    if not isinstance(on_value, relax.PrimValue) or not isinstance(off_value, relax.PrimValue):
        return False

    yield

    on_value = on_value.value.value
    off_value = off_value.value.value

    res = []
    # support 2, 3, 4, 5 dim
    res.append(_check_dim([in_shape], [2, 3, 4, 5], False))
    spec = [32] + [16384, 16384, 16384][4 - dim :]
    res.append(_check_range(in_shape, spec))
    # check depth and on/off value
    res.append(_check_range([attrs.depth], [16384]))
    res.append(_check_range([on_value, off_value], [65535, 65535], 0))
    # check axis
    res.append(-1 <= attrs.axis <= 4)

    return all(res)


@_checker
def _check_argminmax(ctx: PatternCheckContext) -> bool:
    call = ctx.annotated_expr["root"]
    attrs = call.attrs

    if not attrs.keepdims:
        return False

    yield

    in_shape = [int(x) for x in call.args[0].struct_info.shape]
    res = []
    res.append(_check_dim([in_shape], [2, 3, 4, 5]))
    return all(res)


@_checker
def _check_shift(ctx: PatternCheckContext) -> bool:
    call = ctx.annotated_expr["root"]
    in_shape0 = [int(x) for x in call.args[0].struct_info.shape]
    in_shape1 = [int(x) for x in call.args[1].struct_info.shape]

    res = []
    # support 2, 3, 4 dim
    res.append(_check_dim([in_shape0, in_shape1], [2, 3, 4]))
    # Broadcast is not supported, and both inputs should share the same shape
    res.append(in_shape0 == in_shape1)
    res.append(all([i == j for i, j in zip(in_shape0, in_shape1)]))
    res.append(_check_shape(in_shape0, 0, 1, 4))
    return all(res)


@_checker
def _check_scatter_elements(ctx: PatternCheckContext) -> bool:
    call = ctx.annotated_expr["root"]
    in_shape0 = [int(x) for x in call.args[0].struct_info.shape]
    in_shape1 = [int(x) for x in call.args[1].struct_info.shape]
    in_shape2 = [int(x) for x in call.args[2].struct_info.shape]

    attrs = call.attrs

    res = []
    if attrs.reduction == "add":
        res.append(all(x < 16384 for x in in_shape0))
    elif attrs.reduction == "mul":
        res.append(all(x < 4096 for x in in_shape0))
    elif attrs.reduction != "update":
        return False

    res.append(all([i <= j for i, j in zip(in_shape1, in_shape0)]))
    res.append(all([i == j for i, j in zip(in_shape2, in_shape1)]))
    return all(res)


@_checker
def _check_reverse_sequence(ctx: PatternCheckContext) -> bool:
    call = ctx.annotated_expr["root"]
    attrs = call.attrs
    in_shape = [int(x) for x in call.args[0].struct_info.shape]
    out_shape = [int(x) for x in call.struct_info.shape]

    res = []
    # support 2, 3 dim
    res.append(_check_dim([in_shape, out_shape], [2, 3]))
    # check shape of input and output
    res.append(_check_shape(in_shape, 0, 0, 3))
    batch_axis = attrs.batch_axis
    seq_axis = attrs.seq_axis
    res.append(0 <= batch_axis <= 1)
    res.append(0 <= seq_axis <= 1 or (seq_axis == 2 and len(in_shape) == 3))
    res.append(batch_axis != seq_axis)
    return all(res)


def _gen_single_op_pattern(relax_name, num_inputs=1, checker=None):
    pattern = is_op(f"relax.{relax_name}")(*tuple(wildcard() for _ in range(num_inputs)))
    ret = (f"compass.{relax_name}", pattern)
    if checker is not None:
        ret += ({"root": pattern}, checker)
    return ret


def _gen_qnn_single_op_pattern(relax_name, num_inputs=1, checker=None):
    op = is_op(f"relax.{relax_name}")(*tuple(_dq(wildcard()) for _ in range(num_inputs)))
    pattern = _quant(op)
    ret = (f"compass.qnn.{relax_name}", pattern)
    if checker is not None:
        ret += ({"root": op, "out": pattern}, checker)
    return ret


FLOAT_PATTERNS = [
    _convolution2d_pattern(),
    _instance_norm_pattern(),
    _layer_norm_pattern0(),
    _layer_norm_pattern1(),
    _batch_norm_pattern(),
    _hard_swish_pattern(),
    _softplus_pattern(),
    _matmul_add_pattern(),
    _elementwise_relu_pattern(),
    _batch_norm_single_pattern(),
    _dense_pattern(),
    # single op
    _gen_single_op_pattern("nn.pad", checker=_check_pad),
    _gen_single_op_pattern("nn.max_pool2d", checker=_check_max_pool2d),
    _gen_single_op_pattern("nn.avg_pool2d", checker=_check_avg_pool2d),
    _gen_single_op_pattern("clip", 3),
    _gen_single_op_pattern("permute_dims", checker=_check_transpose),
    _gen_single_op_pattern("matmul", 2, checker=_check_matmul),
    _gen_single_op_pattern("sign", checker=_check_sign),
    _gen_single_op_pattern("nn.log_softmax", checker=_check_softmax),
    _gen_single_op_pattern("nn.softmax", checker=_check_softmax),
    _gen_single_op_pattern("nn.lrn", checker=_check_lrn),
    _gen_single_op_pattern("concat", checker=_check_concat),
    _gen_single_op_pattern("clip"),
    _gen_single_op_pattern("exp"),
    _gen_single_op_pattern("log"),
    _gen_single_op_pattern("abs"),
    _gen_single_op_pattern("where", 3),
    _gen_single_op_pattern("scatter_nd", 3, _check_scatter_nd),
    _gen_single_op_pattern("nn.space_to_depth", checker=_check_space_to_depth),
    _gen_single_op_pattern("nn.depth_to_space", checker=_check_depth_to_space),
    _gen_single_op_pattern("reverse_sequence", 2, checker=_check_reverse_sequence),
    # reduce
    _gen_single_op_pattern("prod"),
    _gen_single_op_pattern("sum"),
    _gen_single_op_pattern("variance"),
    _gen_single_op_pattern("mean"),
    _gen_single_op_pattern("min"),
    _gen_single_op_pattern("max"),
    _gen_single_op_pattern("all"),
    _gen_single_op_pattern("any"),
    # binary
    _gen_single_op_pattern("add", 2, _check_add),
    _gen_single_op_pattern("subtract", 2),
    _gen_single_op_pattern("multiply", 2, _check_mul),
    _gen_single_op_pattern("divide", 2),
    _gen_single_op_pattern("minimum", 2),
    _gen_single_op_pattern("maximum", 2),
    _gen_single_op_pattern("left_shift", 2, _check_shift),
    _gen_single_op_pattern("right_shift", 2, _check_shift),
    # act
    _gen_single_op_pattern("nn.relu"),
    _gen_single_op_pattern("nn.silu"),
    _gen_single_op_pattern("tanh"),
    _gen_single_op_pattern("sigmoid"),
    _gen_single_op_pattern("cos"),
    _gen_single_op_pattern("sqrt"),
    _gen_single_op_pattern("erf", checker=_check_erf),
    _gen_single_op_pattern("nn.prelu", 2, checker=_check_prelu),
    # logical
    _gen_single_op_pattern("logical_not"),
    _gen_single_op_pattern("logical_or", 2),
    _gen_single_op_pattern("logical_and", 2),
    _gen_single_op_pattern("logical_xor", 2),
    _gen_single_op_pattern("greater_equal", 2),
    _gen_single_op_pattern("less_equal", 2),
    _gen_single_op_pattern("equal", 2),
    _gen_single_op_pattern("less", 2),
    _gen_single_op_pattern("not_equal", 2),
    _gen_single_op_pattern("greater", 2),
    # others
    _gen_single_op_pattern("reshape", 2),
    _gen_single_op_pattern("negative"),
    _gen_single_op_pattern("image.resize2d", 2, checker=_check_resize2d),
    _gen_single_op_pattern("take", 2),
    _gen_single_op_pattern("strided_slice", 4),
    _gen_single_op_pattern("strided_slice", 5),
    _gen_single_op_pattern("split", checker=_check_split),
    _gen_single_op_pattern("tile"),
    _gen_single_op_pattern("power", 2),
    _gen_single_op_pattern("nn.space_to_batch_nd", checker=_check_s2b_nd),
    _gen_single_op_pattern("nn.batch_to_space_nd", checker=_check_b2s_nd),
    _gen_single_op_pattern("astype"),
    _gen_single_op_pattern("nn.rms_norm", 2, _check_rms_norm),
    _gen_single_op_pattern("one_hot", 3, _check_one_hot),
    _gen_single_op_pattern("argmin", checker=_check_argminmax),
    _gen_single_op_pattern("argmax", checker=_check_argminmax),
    _gen_single_op_pattern("scatter_elements", 3, _check_scatter_elements),
    _gen_single_op_pattern("flip", checker=_check_flip),
    # custom op
    _gen_single_op_pattern("fake_quant_with_min_max_vars"),
    _gen_single_op_pattern("ctc_greedy_decoder", 2, _check_ctc_greedy_decoder),
    _gen_single_op_pattern("decode_box", 6),
    _gen_single_op_pattern("cps_nms", 4),
    _gen_single_op_pattern("channel_shuffle", checker=_check_channel_shuffle),
]


QUANT_PATTERNS = [
    _qnn_conv2d_pattern(),
    _qnn_basic_lstm_pattern(),
    _qnn_matmul_add_pattern(),
    _qnn_hard_swish_pattern(),
    _qnn_squared_difference_pattern(),
    _qnn_elementwise_relu_pattern(),
    _qnn_silu_pattern(),
    _gen_qnn_single_op_pattern("add", 2),
    _gen_qnn_single_op_pattern("subtract", 2),
    _gen_qnn_single_op_pattern("multiply", 2),
    _gen_qnn_single_op_pattern("mean"),
    _gen_qnn_single_op_pattern("rsqrt"),
    _gen_qnn_single_op_pattern("tanh"),
    _gen_qnn_single_op_pattern("sigmoid"),
    _gen_qnn_single_op_pattern("nn.softmax", checker=_check_softmax),
    _gen_qnn_single_op_pattern("nn.prelu", 2, checker=_check_prelu),
    _gen_single_op_pattern("requantize", 5, _check_requantize),
    _gen_qnn_image_resize2d(),
    *_gen_qnn_concat([2, 3, 4, 5]),
    _gen_qnn_single_op_pattern("nn.avg_pool2d", checker=_check_avg_pool2d),
    _gen_qnn_single_op_pattern("nn.max_pool2d", checker=_check_max_pool2d),
    _gen_single_op_pattern("dequantize", 3),
    _gen_single_op_pattern("quantize", 3),
    _gen_qnn_single_op_pattern("matmul", 2, checker=_check_matmul),
]
