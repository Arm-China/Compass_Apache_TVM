# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.
"""Generate .dot file to describe relay IR and generate .png if dot command is installed"""
import tvm
from tvm import relay

# pylint: disable=abstract-method
class DotGen(relay.ExprFunctor):
    """Dot file generator of Relay IR."""

    colors = [
        "antiquewhite",
        "azure",
        "beige",
        "chartreuse",
        "coral",
        "cyan2",
        "darkolivegreen1",
        "darkorchid2",
        "gold",
        "gray",
        "greenyellow",
        "lightblue1",
        "purple",
        "rosybrown1",
    ]

    def __init__(self):
        super().__init__()
        self._dot = ""
        self._var_index = 0
        self._expr_to_name = dict()
        self._local_vars = dict()

    def _get_or_alloc_var_name(self, expr):
        if expr not in self._expr_to_name:
            if isinstance(expr, (relay.Var, relay.GlobalVar)):
                name = expr.name_hint
            elif (
                hasattr(expr, "span") and expr.span is not None and str(expr.span.source_name.name)
            ):
                name = str(expr.span.source_name.name)
            elif isinstance(expr, relay.expr.Constant):
                name = "constant_" + str(self._var_index)
                self._var_index += 1
            else:
                name = f"temp_var_{self._var_index}"
                self._var_index += 1

            name = name.replace("/", "_")
            name = name.replace("-", "_")
            self._expr_to_name[expr] = name

        return self._expr_to_name[expr]

    def gen(self, expr):
        """Generate the dot file for the given Relay IR."""
        self._dot = """
        digraph expr {
        label = \"graph\";
        bgcolor = \"white\";
        node[shape=record];
        """
        self.visit(expr)
        self._dot += """
        }
        """
        return self._dot

    def visit_function(self, func):
        name = self._get_or_alloc_var_name(func)
        self._dot += """
        subgraph %s {
        label = \"%s\";
        """ % (
            name,
            name,
        )
        self.visit(func.body)
        self._dot += """
        }
        """

    def visit_call(self, call):
        for arg in call.args:
            self.visit(arg)
        name = self._get_or_alloc_var_name(call)

        if isinstance(call.op, tvm.ir.op.Op):
            op_type = call.op.name
        else:
            op_type = self._get_or_alloc_var_name(call.op)

        color = hash(op_type) % len(self.colors)
        color = self.colors[color]
        self._dot += """
        %s[label = \"{%s|op : %s}\", style=\"filled\",color=\"black\",fillcolor=\"%s\"];
        """ % (
            name,
            name,
            op_type,
            color,
        )

        for arg in call.args:
            arg_name = self._expr_to_name[arg]
            self._dot += """
            %s -> %s;
            """ % (
                arg_name,
                name,
            )

    def visit_tuple_getitem(self, expr):
        self.visit(expr.tuple_value)
        name = self._get_or_alloc_var_name(expr)
        self._dot += """
        %s[label = \"get tuple item %s\"];
        """ % (
            name,
            str(expr.index),
        )

        tuple_name = self._expr_to_name[expr.tuple_value]

        self._dot += """
        %s -> %s;
        """ % (
            tuple_name,
            name,
        )

    def visit_var(self, expr):
        name = self._get_or_alloc_var_name(expr)
        if expr not in self._local_vars:
            self._dot += """
            %s[label = \"%s\"];
            """ % (
                name,
                name,
            )
        else:
            raise NotImplementedError()

    def visit_constant(self, expr):
        name = self._get_or_alloc_var_name(expr)
        self._dot += """
        %s[label = \"%s\", style=\"filled\",color=\"black\",fillcolor=\"greenyellow\"];
        """ % (
            name,
            name,
        )

    def visit_let(self, expr):
        var = expr.var
        value = expr.value
        self._local_vars[var] = value
        self.visit(expr.body)
        body_name = self._expr_to_name[expr.body]

        self._expr_to_name[expr] = body_name

    def visit_tuple(self, expr):
        for arg in expr.fields:
            self.visit(arg)
        name = self._get_or_alloc_var_name(expr)
        self._dot += """
        %s[label = \"make tuple\";color=\"black\";fillcolor=\"firebrick2\"];
        """ % (
            name
        )

        for arg in expr.fields:
            arg_name = self._expr_to_name[arg]
            self._dot += """
            %s -> %s;
            """ % (
                arg_name,
                name,
            )

    def visit_if(self, expr):
        true_branch = expr.true_branch
        false_branch = expr.false_branch
        cond = expr.cond

        t_name = self._get_or_alloc_var_name(true_branch)
        f_name = self._get_or_alloc_var_name(false_branch)

        t_branch_name = t_name + "_T"
        f_branch_name = f_name + "_F"

        cond_name = self._get_or_alloc_var_name(cond)
        if_name = self._get_or_alloc_var_name(expr)
        if_begin_name = if_name + "_begin"

        self._dot += (
            """
        subgraph %s {
        compound=true;
        """
            % if_begin_name
        )

        self._dot += (
            """
        subgraph %s {
        color=\"blue\";
        label = \"True\";
        """
            % t_branch_name
        )
        self.visit(true_branch)
        self._dot += """
        }
        """

        self._dot += (
            """
        subgraph %s {
        color=\"red\";
        label = \"False\";
        """
            % f_branch_name
        )
        self.visit(false_branch)
        self._dot += """
        }
        """
        self.visit(cond)

        self._dot += """
        %s -> %s [lheader=%s, label = \"True\"];
        %s -> %s [lheader=%s, label = \"False\"];
        """ % (
            cond_name,
            t_name,
            t_branch_name,
            cond_name,
            f_name,
            f_branch_name,
        )
        self._dot += (
            """
        %s[label = \"if end\";color=\"black\";fillcolor=\"aquamarine4\"];
        """
            % if_name
        )

        self._dot += """
        %s -> %s;
        %s -> %s;
        """ % (
            t_name,
            if_name,
            f_name,
            if_name,
        )
        self._dot += """
        }
        """


def to_dot(expr):
    """generate .dot script based on relay IR for visualization usage

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression

    Returns
    -------
    source : str
        the generated .dot script
    """
    if isinstance(expr, tvm.ir.module.IRModule):
        expr = expr["main"]
    source = DotGen().gen(expr)
    return source
