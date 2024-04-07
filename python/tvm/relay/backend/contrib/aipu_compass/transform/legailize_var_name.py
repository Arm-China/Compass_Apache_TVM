# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""Legalize var name so that it could be parsed by text parser"""
from tvm import relay


def verify_name(name):
    name = name.replace("-", "_")
    name = name.replace(".", "_")
    name = name.replace(":", "_")
    return name


class VarNameRewriter(relay.ExprMutator):
    """Legalize var name"""

    def visit_var(self, var):
        name = var.name_hint
        new_name = verify_name(name)
        if new_name != name:
            return relay.Var(new_name, var.type_annotation)
        return var


@relay.transform.function_pass(opt_level=0, name="LegalizeVarName")
class LegalizeVarName(relay.ExprMutator):
    """legalize var name to rewrite illegal character in var name"""

    def transform_function(self, func, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        return VarNameRewriter().visit(func)
