# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""Set specify nodes target compiler with default."""
import tvm
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass import analysis


class InlineFunction(relay.ExprMutator):
    """Inline Function to Call and replace params to args."""

    def __init__(self):
        super().__init__()
        self.param2args = {}

    def visit_call(self, call):
        ret = super().visit_call(call)
        new_args = []
        for arg in ret.args:
            if isinstance(arg, relay.Var) and arg in self.param2args.keys():
                new_args.append(self.param2args[arg])
            else:
                if not isinstance(arg, relay.Constant):
                    arg = relay.annotation.compiler_end(arg, "default")
                new_args.append(relay.annotation.compiler_begin(arg, "default"))
        return relay.Call(ret.op, new_args, ret.attrs, ret.type_args, ret.span)

    def __call__(self, func, args):
        assert len(args) == len(func.params)
        for i in range(len(args)):
            self.param2args[func.params[i]] = args[i]

        new_func = self.visit(func)
        return new_func.body


class Rewriter(relay.ExprMutator):
    """Rewrite compiler default to compiler_begin/end of each exclude_nodes."""

    def __init__(self, exclude_nodes):
        super().__init__()
        self.exclude_nodes = exclude_nodes

    def visit_call(self, call):
        ret = super().visit_call(call)
        if ret.op == relay.op.get("annotation.compiler_end"):
            inp = ret.args[0]
            if inp not in self.exclude_nodes:
                return ret
            if isinstance(inp, relay.Call):
                new_begins = []
                for begin in inp.args:
                    assert begin.op == relay.op.get("annotation.compiler_begin")
                    new_begin = relay.annotation.compiler_begin(begin.args[0], "default")
                    new_begins.append(new_begin)
                if isinstance(inp.op, relay.Function):
                    new_inp = InlineFunction()(inp.op, new_begins)
                else:
                    new_inp = relay.Call(inp.op, new_begins, inp.attrs, inp.type_args, inp.span)
                new_end = relay.annotation.compiler_end(new_inp, "default")
                return new_end
            elif isinstance(inp, relay.Tuple):
                new_begins = []
                for field in inp.fields:
                    assert field.op == relay.op.get("annotation.compiler_begin")
                    new_begin = relay.annotation.compiler_begin(field.args[0], "default")
                    new_begins.append(new_begin)
                new_tup = relay.Tuple(new_begins, inp.span)
                new_end = relay.annotation.compiler_end(new_tup, "default")
                return new_end
            elif isinstance(inp, relay.TupleGetItem):
                begin = inp.tuple_value
                assert begin.op == relay.op.get("annotation.compiler_begin")
                new_begin = relay.annotation.compiler_begin(begin.args[0], "default")
                new_inp = relay.TupleGetItem(new_begin, inp.index, inp.span)
                new_end = relay.annotation.compiler_end(new_inp, "default")
                return new_end
            else:
                assert False, "Need to support."

        return ret

    def __call__(self, mod):
        new_func = self.visit(mod["main"])
        mod.update_func(mod.get_global_var("main"), new_func)
        return mod


@tvm.ir.transform.module_pass(opt_level=0)
class SetNodeCompilerToDefault:
    """
    Annotate nodes' compiler to default depend on given index.

    Parameters
    ----------
    indices: list of int
        The indices of nodes in relay ir to annotate to cpu. Need to
        specify node upon compiler_end.
    """

    def __init__(self, indices=None):
        self.indices = indices

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """try transform module"""
        if self.indices is None:
            return mod
        exclude_nodes = [analysis.printer_index_to_expr(mod)[node_id] for node_id in self.indices]
        update_mod = Rewriter(exclude_nodes)(mod)

        return relay.transform.InferType()(update_mod)
