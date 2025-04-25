# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Lower tag expr and optimize its node by its attr."""
from tvm import tir, DataType
from .utils import is_builtin
from ...logger import WARN


class _Mutator(tir.StmtExprMutator):
    def __init__(self, aipu_info):
        super().__init__()
        self._aipu_info = aipu_info

    def _mutate_tag(self, call):
        _, node, attr = call.args

        def _del_tag():
            WARN(f'This "S.tag" will not work, please check the usage of "S.tag" here: {call}')
            return node

        if is_builtin(node, "vcast") and attr == "no_zip":
            if not self._aipu_info.is_v2:
                return _del_tag()

            part = node.args[1]
            inp = node.args[3]
            from_dtype = DataType(inp.dtype).element_of
            to_dtype = DataType(node.dtype).element_of
            if part != "all" or from_dtype != "float16" or to_dtype != "float32":
                return _del_tag()

            node_args = list(node.args)
            node_args[1] = "all_with_no_zip"
            return tir.Call(node.dtype, node.op, node_args, node.span)
        return _del_tag()

    def visit_call(self, call):
        ret = super().visit_call(call)
        if is_builtin(ret, "tag"):
            return self._mutate_tag(ret)
        return ret


@tir.transform.prim_func_pass(opt_level=0)
class LowerTag:
    """Lower tag expr and optimize its node by its attr."""

    def __init__(self, aipu_info):
        self._aipu_info = aipu_info

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given PrimFunc, transform it and return the result."""
        new_body = _Mutator(self._aipu_info).visit(func.body)
        return func.with_body(new_body, func.span)
