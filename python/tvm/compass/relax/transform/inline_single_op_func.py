# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Inline single op sub func in partitoned function."""
from tvm import relax
from tvm.relax.expr_functor import PyExprMutator, mutator
from .utils import is_call, is_compass_func


@mutator
class Inliner(PyExprMutator):
    """Inline single op func mutator."""

    def __init__(self, var2val, mod=None):
        super().__init__(mod)
        self.var2val = var2val
        self.func2value_info = {}

    def visit_function_(self, func):
        skip_patterns = ("batch_norm_single", "dense")
        if not (
            "Composite" in func.attrs
            and func.attrs["Composite"].startswith("compass")
            and func.attrs["Composite"].split(".")[-1] not in skip_patterns
            and len(func.body.blocks) == 1
            and len(func.body.blocks[0].bindings) == 1
        ):
            return super().visit_function_(func)
        value = func.body.blocks[0].bindings[0].value
        index_mapping = {}
        params = list(func.params)
        if isinstance(value, relax.Call):
            value_args = value.args[0].fields if is_call(value, "concat") else value.args
            for i, arg in enumerate(value_args):
                if arg in params:
                    index_mapping[i] = params.index(arg)
        elif isinstance(value, relax.TupleGetItem):
            index_mapping["tup_get_item_index"] = value.index
        else:
            assert isinstance(value, relax.Tuple)
            for i, item in enumerate(value):
                if item in params:
                    index_mapping[i] = params.index(item)
        self.func2value_info[func] = (value, index_mapping)
        return func

    def visit_call_(self, call):
        if call.op in self.var2val and self.var2val[call.op] in self.func2value_info:
            value_info = self.func2value_info[self.var2val[call.op]]
            value, index_mapping = value_info
            if isinstance(value, relax.Call):
                args = value.args[0].fields if is_call(value, "concat") else value.args
                real_args = list(args)
                for i, j in index_mapping.items():
                    real_args[i] = call.args[j]
                real_args = [relax.Tuple(real_args)] if is_call(value, "concat") else real_args
                ret = relax.Call(value.op, real_args, value.attrs)
            elif isinstance(value, relax.TupleGetItem):
                ret = relax.TupleGetItem(call.args[0], index_mapping["tup_get_item_index"])
            else:
                assert isinstance(value, relax.Tuple)
                real_fields = list(value.fields)
                for i, j in index_mapping.items():
                    real_fields[i] = call.args[j]
                assert isinstance(value, relax.Tuple)
                ret = relax.Tuple(real_fields)
            return ret
        return super().visit_call_(call)


@relax.transform.function_pass(opt_level=0)
class InlineSingleOpFunc:
    """Inline single op sub func in partitoned function."""

    def transform_function(self, func, mod, pass_ctx):  # pylint: disable=unused-argument
        """Transform the given function and return the result."""
        if not is_compass_func(func):
            return func
        updated_func = Inliner(relax.analysis.get_var2val(func)).visit_expr(func)
        updated_func = relax.analysis.remove_all_unused(updated_func)
        return updated_func
