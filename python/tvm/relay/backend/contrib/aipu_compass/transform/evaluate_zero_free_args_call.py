# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""Evaluate non primitive call that all args are constant."""
import tvm
from tvm import relay


def vmobj_to_list(obj):
    """help function to convert ADT structure to ndarray"""

    def to_list(obj):
        return obj if isinstance(obj, list) else [obj]

    if isinstance(obj, tvm.nd.NDArray):
        return obj
    if isinstance(obj, tvm.runtime.container.ADT):
        result = []
        for fun in obj:
            result.extend(to_list(vmobj_to_list(fun)))
        return result
    if isinstance(obj, tvm.relay.backend.interpreter.ConstructorValue):
        if obj.constructor.name_hint == "Cons":
            tl_ = to_list(vmobj_to_list(obj.fields[1]))
            hd_ = to_list(vmobj_to_list(obj.fields[0]))
            hd_.extend(tl_)
            return hd_
        if obj.constructor.name_hint == "Nil":
            return []
        if "tensor_nil" in obj.constructor.name_hint:
            return [0]
        if "tensor" in obj.constructor.name_hint:
            return [obj.fields[0]]
        raise RuntimeError("Unknown object type: %s" % obj.constructor.name_hint)
    raise RuntimeError("Unknown object type: %s" % type(obj))


def ndarrays_to_expr(datas):
    """help function to convert numpy ndarray to expr"""
    if isinstance(datas, list):
        converted = [ndarrays_to_expr(data) for data in datas]
        return relay.Tuple(converted)
    if isinstance(datas, tvm.nd.NDArray):
        return relay.Constant(datas)
    raise RuntimeError("Unknown object type: %s" % type(datas))


class EvaluateZeroFreeArgsCallMutator(relay.ExprMutator):
    """Evaluate non primitive call that all args are constant.(evalutate ADT function)"""

    def visit_call(self, call):
        call = super().visit_call(call)
        if all([isinstance(arg, relay.expr.Constant) for arg in call.args]) or len(call.args) == 0:
            result = relay.create_executor(kind="vm").evaluate(call)
            datas = vmobj_to_list(result)
            return ndarrays_to_expr(datas)
        return call


@relay.transform.function_pass(opt_level=0, name="EvaluateZeroFreeArgsCall")
class EvaluateZeroFreeArgsCall:
    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        return EvaluateZeroFreeArgsCallMutator().visit(func)
