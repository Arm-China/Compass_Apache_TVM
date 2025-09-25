# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Convert ops in Compass subfunc."""
import numpy as np
from tvm import relax, ir
from tvm.relax.expr_functor import PyExprMutator, mutator
from .utils import is_compass_func


def _broadcast_inputs(call):
    shapes = [arg.struct_info.shape for arg in call.args[:2]]
    try:
        shapes = [[int(idx) for idx in shape] for shape in shapes]
    except TypeError:
        return None
    max_dim = max([len(shape) for shape in shapes])
    origin_args = list(call.args[:2])
    args = []
    new_shapes = []
    for idx, shape in enumerate(shapes):
        arg = origin_args[idx]
        new_shape = shape
        if len(shape) != max_dim:
            new_shape = [1] * (max_dim - len(shape)) + shape
            if isinstance(arg, relax.Constant):
                data = arg.data.numpy()
                arg = relax.const(data.reshape(new_shape), data.dtype)
            else:
                arg = relax.op.reshape(arg, new_shape)
        args.append(arg)
        new_shapes.append(new_shape)

    new_shape = []
    for dim in range(max_dim):
        cur_dim_shape = [shape[dim] for shape in new_shapes]
        dim_max_size = max(cur_dim_shape)
        new_shape.append(dim_max_size)

    new_args = []
    for arg, shape in zip(args, new_shapes):
        new_arg = arg
        if shape != new_shape:
            reps = [dim0 // dim1 for dim0, dim1 in zip(new_shape, shape)]
            if isinstance(arg, relax.Constant):
                data = arg.data.numpy()
                new_arg = relax.const(np.tile(data, reps), data.dtype)
            else:
                new_arg = relax.op.tile(arg, reps)
        new_args.append(new_arg)
    if new_args == origin_args:
        return call
    new_args += call.args[2:]
    return relax.Call(call.op, new_args, call.attrs)


@mutator
class Convertor(PyExprMutator):
    """Convert ops to equivalent but more efficient ops."""

    def visit_function_(self, func):
        # Avoid visiting call in composite functions.
        if "Composite" in func.attrs and func.attrs["Composite"].startswith("compass"):
            return func
        return super().visit_function_(func)

    def visit_call_(self, call):
        ret = super().visit_call_(call)
        _broadcast_ops = [
            "add",
            "multiply",
            "subtract",
            "divide",
            "maximum",
            "minimum",
            "logical_and",
            "logical_or",
            "logical_xor",
            "equal",
            "less",
            "greater_equal",
            "less_equal",
            "greater",
            "not_equal",
        ]
        if not isinstance(ret.op, ir.Op) or ret.op.name[6:] not in _broadcast_ops:
            return ret
        shape1 = ret.args[0].struct_info.shape.values
        shape2 = ret.args[1].struct_info.shape.values
        if ir.structural_equal(shape1, shape2):
            return ret
        new_call = _broadcast_inputs(ret)
        return new_call if new_call is not None else ret


@relax.transform.function_pass(opt_level=0)
class ConvertCompassOps:
    """Convert ops in Compass subfunc."""

    def transform_function(self, func, mod, pass_ctx):  # pylint: disable=unused-argument
        """Transform the given function and return the result."""
        if not is_compass_func(func):
            return func
        return Convertor().visit_expr(func)
