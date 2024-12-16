# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Make the IR simpler and more efficient through some equivalent transformations."""
import math
from tvm import ir, tir, DataType
from ... import script as S
from .utils import is_broadcast_const


def _is_power_of_two_integer_const(x):
    if is_broadcast_const(x) and DataType(x.dtype).is_integer:
        n = x.args[2].value
        return n > 0 and (n & (n - 1)) == 0

    return False


class _Mutator(tir.StmtExprMutator):
    def _mutate_vmul(self, call):
        r, x, y, mask = call.args[1:]

        # Optimize multiplication to left shift.
        if not _is_power_of_two_integer_const(y):
            return call

        shift_value = int(math.log2(y.args[2].value))
        return S.vsl(x, shift_value, mask, r=r)

    def visit_call(self, call):
        ret = super().visit_call(call)

        if ret.op != ir.Op.get("tir.call_extern"):
            return ret

        func_name = ret.args[0].value
        if func_name == "vmul":
            return self._mutate_vmul(ret)

        return ret


@tir.transform.prim_func_pass(opt_level=0)
class Simplify:
    """Make the IR simpler and more efficient through some equivalent transformations."""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        return func.with_body(_Mutator().visit(func.body), span=func.span)
