# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=unused-argument, unsupported-binary-operation
"""Relax IR to Compass IR mapping rules."""
import inspect
from functools import reduce, wraps
from tvm.tir import Any
from operator import mul
from typing import List, Tuple, Union

from tvm.ir.container import Array
from tvm.tir import Any
from tvm import relax, ir, tir
from tvm.relax.dpl import (
    is_const,
    is_op,
    is_shape,
    wildcard,
    is_var,
    is_tuple_get_item,
    is_tuple,
)
from tvm.relax.transform import PatternCheckContext
from tvm.relax.backend.utils import has_leaking_intermediate_variables
from tvm.relax.backend.contrib.aipu_compass import AipuCompassConfig


def _convert_array(arr):
    if isinstance(arr, (tir.IntImm, int)):
        return int(arr)
    if isinstance(arr, (list, Tuple, Array)):
        return [_convert_array(val) for val in arr]
    return int(arr)


def _check_range(
    check_list: Union[List[int], Tuple[int], Array], rule: List[int], min_value=1
) -> bool:
    if len(check_list) != len(rule):
        return False
    # ensure that shape should not be dynamic.
    if isinstance(check_list, (list, Tuple, Array)) and _is_dynamic(check_list):
        return False

    return all(min_value <= x <= y for x, y in zip(_convert_array(check_list), rule))


def _check_shape(
    shape: Union[List[int], Tuple[int]], shape_0_idx: int, shape_n_idx: int, largest_dim: int
) -> bool:
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


def _check_dim(shapes: Union[List[Array], Tuple[Array]], dims: List[int], keep_dims=True) -> bool:
    shape_dims = set(len(shape) for shape in shapes)
    if not shape_dims.issubset(set(dims)):
        return False
    if keep_dims:
        return len(shape_dims) == 1
    return True


def unpack_commutative_args(call, rhs_name="const"):
    """Unpack arguments of the binary operators consider commutative, ensure the
    right hand side operand is the expected one."""
    assert isinstance(call, relax.Call)
    ops = (
        relax.op.add,
        relax.op.multiply,
        relax.op.maximum,
        relax.op.minimum,
    )

    lhs, rhs = call.args
    if rhs_name == "const":
        if isinstance(rhs, relax.Constant):
            return lhs, rhs
        assert call.op in ops
        assert isinstance(lhs, relax.Constant)
        return rhs, lhs

    if isinstance(rhs, relax.Call) and rhs.op == getattr(relax.op, rhs_name):
        return lhs, rhs

    assert call.op in ops
    assert isinstance(lhs, relax.Call) and lhs.op == getattr(relax.op, rhs_name)
    return rhs, lhs


def _check_conv2d(conv2d: relax.Call, add_const: relax.Constant):
    # Check if it is supported by AIPU Compass.
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
    def _get_shapes(ir_type):
        if isinstance(ir_type, ir.TensorType):
            return [ir_type.shape]
        if isinstance(ir_type, ir.TupleType):
            shapes = []
            for field in ir_type.fields:
                shapes += _get_shapes(field)
            return shapes
        return []

    shapes = _get_shapes(shape_or_type) if isinstance(shape_or_type, ir.Type) else [shape_or_type]
    for shape in shapes:
        if any(isinstance(dim, Any) for dim in shape):
            return True
    return False


def _dq(x):
    return is_op("relax.dequantize")(x, is_const(), is_const())


def _q(x):
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
        if AipuCompassConfig.get().common["disable_op_spec_checker"] == "true":
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
    def check(ctx: PatternCheckContext) -> bool:
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

    return ("aipu_compass.conv2d", pattern, annotations, check)


def _qnn_conv2d_pattern():
    conv = is_op("relax.nn.conv2d")(_dq(wildcard()), _dq(is_const()))
    add_const = is_const()
    conv_add = is_op("relax.add")(conv, _dq(add_const))
    conv_add_relu = is_op("relax.nn.relu")(conv_add)
    conv_add_leaky_relu = is_op("relax.nn.leakyrelu")(conv_add)
    conv_add_clip = is_op("relax.clip")(conv_add).has_attr({"a_min": 0.0, "a_max": 6.0})
    conv_leaky_relu = is_op("relax.nn.leakyrelu")(conv)
    conv_relu = is_op("relax.nn.relu")(conv)

    pattern = (
        _q(conv)
        | _q(conv_add)
        | _q(conv_add_relu)
        | _q(conv_add_leaky_relu)
        | _q(conv_add_clip)
        | _q(conv_leaky_relu)
        | _q(conv_relu)
    )

    annotations = {"root": pattern, "conv": conv, "add_const": add_const}

    @_checker
    def check(ctx: PatternCheckContext) -> bool:
        if has_leaking_intermediate_variables(ctx):
            return False

        conv2d = ctx.annotated_expr.get("conv")
        add_const = ctx.annotated_expr.get("add_const")
        return _check_conv2d(conv2d, add_const)

    return ("aipu_compass.qnn.conv2d", pattern, annotations, check)


def _batchnorm_pattern():
    multiply = is_op("multiply")(wildcard(), is_const())
    out = is_op("add")(multiply, is_const())
    annotations = {"root": out}

    @_checker
    def check(ctx: PatternCheckContext) -> bool:
        if has_leaking_intermediate_variables(ctx):
            return False

        # Check if the given match is supported by AIPU Compass.
        batchnorm = ctx.annotated_expr["root"]
        mul_in, const_in = unpack_commutative_args(batchnorm)
        shape = [int(val) for val in const_in.struct_info.shape.values]
        if len(shape) != 1 and not all([val == 1 for val in shape[:-1]]):
            return False

        _, const_in = unpack_commutative_args(mul_in)
        shape = [int(val) for val in const_in.struct_info.shape.values]
        if len(shape) != 1 and not all([val == 1 for val in shape[:-1]]):
            return False

        yield  # The Compass OP Spec check code must be placed after this statement.

        return True

    return ("aipu_compass.batch_norm", out, annotations, check)


def _batchnorm_arith_pattern():
    def _make_batchnorm_arith_pattern(op):
        assert op in ("add", "multiply", "subtract", "divide")
        out = is_op(op)(wildcard(), is_const())
        annotations = {"root": out}
        return "aipu_compass.batch_norm", out, annotations

    @_checker
    def check(ctx: PatternCheckContext) -> bool:
        # Check if the given match is supported by AIPU Compass.
        batch_norm = ctx.annotated_expr["root"]
        arg_in, const_in = unpack_commutative_args(batch_norm)
        if isinstance(arg_in, relax.Constant):
            return False
        if len(arg_in.struct_info.shape) not in (3, 4):
            return False
        shape = [int(val) for val in const_in.struct_info.shape.values]
        if len(shape) != 1 and not all([val == 1 for val in shape[:-1]]):
            return False

        yield  # The Compass OP Spec check code must be placed after this statement.
        # for quantinize convinent, also bn is faster than add/mul/sub
        return True

    return [
        (*_make_batchnorm_arith_pattern("add"), check),
        (*_make_batchnorm_arith_pattern("multiply"), check),
        (*_make_batchnorm_arith_pattern("subtract"), check),
        (*_make_batchnorm_arith_pattern("divide"), check),
    ]


def _batchnorm_single_pattern():
    inputs = (wildcard(),) + tuple(is_const() for _ in range(4))
    out = is_op("relax.nn.batch_norm")(*inputs)
    return ("aipu_compass.batch_norm", out)


def _instance_norm_pattern():
    inp = wildcard()
    mean = is_op("relax.mean")(inp)
    sub = is_op("relax.subtract")(inp, mean)
    var = is_op("relax.variance")(inp)
    add = is_op("relax.add")(var, is_const())
    sqrt = is_op("relax.sqrt")(add)
    divide = is_op("relax.divide")(sub, sqrt)
    mul = is_op("relax.multiply")(divide, is_const())
    out = is_op("relax.add")(mul, is_const())
    annotations = {"root": out, "input": inp}

    @_checker
    def check(ctx: PatternCheckContext) -> bool:
        in_shape = ctx.annotated_expr["input"].struct_info.shape.values
        out_shape = ctx.annotated_expr["root"].struct_info.shape.values

        res = []
        # only support 4-dim
        res.append(_check_dim([in_shape, out_shape], [4]))
        # check shape of input and output
        res.append(_check_range(in_shape[1:], [16384, 16384, 16384]))
        res.append(_check_range(out_shape[1:], [16384, 16384, 16384]))
        return all(res)

    return ("aipu_compass.instance_norm", out, annotations, check)


def _matmul_pattern():
    out = is_op("relax.matmul")(wildcard(), wildcard())
    annotations = {"root": out}

    @_checker
    def check(ctx: PatternCheckContext) -> bool:
        if has_leaking_intermediate_variables(ctx):
            return False

        matmul = ctx.annotated_expr["root"]
        in_shape1 = [int(val) for val in matmul.args[0].struct_info.shape.values]
        in_shape2 = [int(val) for val in matmul.args[1].struct_info.shape.values]

        res = []
        # check shape of input and output
        res.append(_check_range(in_shape1, [16384] * 2))
        res.append(_check_range(in_shape2, [16384] * 2))
        return all(res)

    return ("aipu_compass.matmul", out, annotations, check)


def _pad_pattern():
    out = is_op("relax.nn.pad")(wildcard(), is_const())
    annotations = {"root": out}

    @_checker
    def check(ctx: PatternCheckContext) -> bool:
        # Check if it is supported by AIPU Compass.
        pad = ctx.annotated_expr["root"]
        # Check pad_mod.
        attrs = pad.attrs
        if attrs.pad_mode not in ["constant", "reflect"]:
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
        res.append(len(pad_width) == 8)
        res.append(_check_range(pad_width, [128] * 8, -1))
        return all(res)

    return ("aipu_compass.pad", out, annotations, check)


def _elementwise_relu_pattern():
    pattern_op = is_op("relax.add") | is_op("relax.subtract") | is_op("relax.multiply")
    pattern = pattern_op(wildcard(), wildcard())
    pattern = is_op("relax.nn.relu")(pattern)

    return ("aipu_compass.eltwise_relu", pattern)


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
    pattern = is_op("relax.add")(matmul, add_const)
    annotations = {"root": pattern, "add_const": add_const, "inp": inp}

    @_checker
    def check(ctx: PatternCheckContext) -> bool:
        # Check if the given match is supported by AIPU Compass.
        inp = ctx.annotated_expr["inp"]
        add_const = ctx.annotated_expr["add_const"]
        return _check_matmul_add(inp, add_const)

    return ("aipu_compass.matmul_add", pattern, annotations, check)


def _qnn_matmul_add_pattern():
    inp = wildcard()
    matmul = is_op("relax.matmul")(_dq(inp), _dq(is_const()))
    add_const = is_const()
    add = is_op("relax.add")(matmul, _dq(add_const))
    pattern = _q(add)
    annotations = {"root": pattern, "inp": inp, "add_const": add_const}

    @_checker
    def check(ctx: PatternCheckContext) -> bool:
        # Check if the given match is supported by AIPU Compass.
        inp = ctx.annotated_expr["inp"]
        add_const = ctx.annotated_expr["add_const"]
        return _check_matmul_add(inp, add_const)

    return ("aipu_compass.qnn.matmul_add", pattern, annotations, check)


def _transpose_pattern():
    pattern = is_op("relax.permute_dims")(wildcard())
    annotations = {"root": pattern}

    @_checker
    def check(ctx: PatternCheckContext) -> bool:
        transpose = ctx.annotated_expr["root"]
        in_shape = transpose.args[0].struct_info.shape.values
        res = []
        # support 1, 2, 3, 4, 5, 6 dim
        res.append(_check_dim([in_shape], [1, 2, 3, 4, 5, 6]))
        return all(res)

    return ("aipu_compass.transpose", pattern, annotations, check)


def _max_pool2d_pattern():
    out = is_op("relax.nn.max_pool2d")(wildcard())
    annotations = {"root": out}

    @_checker
    def check(ctx: PatternCheckContext) -> bool:
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
        res.append(
            all(p < kernels[0] for p in pads[::2]) and all(p < kernels[1] for p in pads[1::2])
        )

        return all(res)

    return ("aipu_compass.max_pool2d", out, annotations, check)


def _reduce_pattern():
    def _make_reduce_pattern(method):
        out = is_op(f"relax.{method}")(wildcard())
        annotations = {"root": out}
        return f"aipu_compass.{method}", out, annotations

    @_checker
    def check(ctx: PatternCheckContext) -> bool:
        call = ctx.annotated_expr["root"]
        # Check if it is supported by AIPU Compass.
        if not call.attrs.keepdims:
            return False

        yield  # The Compass OP Spec check code must be placed after this statement.

        return True

    return [
        (*_make_reduce_pattern("prod"), check),
        (*_make_reduce_pattern("sum"), check),
        (*_make_reduce_pattern("variance"), check),
        (*_make_reduce_pattern("mean"), check),
        (*_make_reduce_pattern("min"), check),
        (*_make_reduce_pattern("max"), check),
    ]


def _sign_pattern():
    out = is_op(f"relax.sign")(wildcard())
    annotations = {"root": out}

    @_checker
    def check(ctx: PatternCheckContext) -> bool:
        call = ctx.annotated_expr["root"]
        in_shape0 = call.args[0].struct_info.shape.values

        res = []
        res.append(_check_dim([in_shape0], [2, 3, 4]))
        res.append(_check_shape(in_shape0, 0, 0, 4))
        return all(res)

    return (f"aipu_compass.sign", out, annotations, check)


def _concatenate_pattern():
    out = is_op(f"relax.concat")(wildcard())
    annotations = {"root": out}

    @_checker
    def check(ctx: PatternCheckContext) -> bool:
        call = ctx.annotated_expr["root"]
        res = []
        res.append(len(call.args[0]) >= 2)
        return all(res)

    return (f"aipu_compass.concat", out, annotations, check)


def _gen_pattern_no_check(compass_name, relax_name, num_inputs):
    pattern = is_op(f"relax.{relax_name}")(*tuple(wildcard() for _ in range(num_inputs)))
    return (f"aipu_compass.{compass_name}", pattern)


def _gen_qnn_pattern_no_check(compass_name, relax_name, num_inputs):
    op = is_op(f"relax.{relax_name}")(*tuple(_dq(wildcard()) for _ in range(num_inputs)))
    pattern = _q(op)
    return (f"aipu_compass.qnn.{compass_name}", pattern)


def pattern_table_pre():
    """
    The AIPU Compass pattern table.
    Should be composited before aipu op convert
    """
    float_patterns = [
        _convolution2d_pattern(),
        _max_pool2d_pattern(),
        _batchnorm_pattern(),
        *_batchnorm_arith_pattern(),
        _batchnorm_single_pattern(),
        _instance_norm_pattern(),
        *_reduce_pattern(),
        _matmul_add_pattern(),
        _elementwise_relu_pattern(),
        # single op with check
        _transpose_pattern(),
        _matmul_pattern(),
        _pad_pattern(),
        _sign_pattern(),
        _concatenate_pattern(),
        # single op without check
        _gen_pattern_no_check("clip", "clip", 3),
        # binary
        _gen_pattern_no_check("add", "add", 2),
        _gen_pattern_no_check("subtract", "subtract", 2),
        _gen_pattern_no_check("multiply", "multiply", 2),
        _gen_pattern_no_check("divide", "divide", 2),
        _gen_pattern_no_check("minimum", "minimum", 2),
        _gen_pattern_no_check("maximum", "maximum", 2),
        # act
        _gen_pattern_no_check("relu", "nn.relu", 1),
        _gen_pattern_no_check("tanh", "tanh", 1),
        # others
        _gen_pattern_no_check("reshape", "reshape", 2),
    ]
    quant_patterns = [
        _qnn_conv2d_pattern(),
        _qnn_matmul_add_pattern(),
        _gen_qnn_pattern_no_check("add", "add", 2),
        _gen_qnn_pattern_no_check("mean", "mean", 1),
    ]
    return quant_patterns + float_patterns


def pattern_table_post(include_float, include_quant):
    """
    The AIPU Compass pattern table.
    Composited after aipu op convert
    """
    float_patterns = []
    quant_patterns = []
    ret = []
    if include_float:
        ret += float_patterns
    if include_quant:
        ret += quant_patterns
    return ret
