# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""TF: Tensorflow frontend."""
import warnings
from collections import defaultdict
import numpy as np
import tvm
from tvm import relax, tir
from tvm.relax.expr_functor import PyExprMutator, mutator
from .tensorflow_ops import _convert_map


__all__ = ["from_tensorflow"]


# An internal list to contain all the control flow primitives used in Tensorflow
# 1.x.
_control_flow_nodes = ["Merge", "Switch", "NextIteration", "Exit", "Enter", "LoopCond"]

# A map to record tensor array write ops and input ta/tensor indices
# Value is (index of tensor array, index of written node)
_tensor_array_write_ops = {
    "TensorArrayWrite": (3, 2),
    "TensorArrayScatter": (0, 2),
    "TensorArraySplit": (0, 1),
}


def is_tensor_array_constuctor(tf_node):
    """Check whether is tensor array constructor node."""
    is_ta = False
    ta_start = "TensorArrayV"
    if tf_node.op.startswith(ta_start):
        is_ta = tf_node.op[len(ta_start)].isnumeric()
    return is_ta


def find_parent_loop_name(node_name, while_loop_name_set):
    """Find name of direct parent while loop."""
    ploop_name = ""
    name_prefix = node_name.rsplit("/", 1)[0]
    if name_prefix.startswith("^"):
        name_prefix = name_prefix[1:]
    for lname in while_loop_name_set:
        if name_prefix.startswith(lname) and len(ploop_name) < len(lname):
            ploop_name = lname

    if len(ploop_name) == 0:
        ploop_name = name_prefix

    return ploop_name


def _in_while_loop(control_flow_node_map, op_name):
    """
    Check if a given control flow operator is part of a while loop execution
    frame. This is based on the fact that there is only one occurrence of
    `LoopCond` for a loop execution frame and it is only presented in the loop
    construct.

    Parameters
    ----------
    control_flow_node_map : Dict[str, Set[str]]
        A dictionary contains the unique control flow execution frame name to
        a set of primitive operators mapping.

    op_name : str
        The name of a control flow primitive.

    Returns
    -------
    ret : bool
        Return true if the operator is in a while loop execution frame,
    otherwise, return false.
    """
    return op_name in control_flow_node_map and "LoopCond" in control_flow_node_map[op_name]


def _get_more_static_shape(shape0, shape1):
    """Compare two shapes with the same rank,
    and return the one with fewer symbolic dimension.
    """
    assert len(shape0) == len(shape1)
    num_sym_dim0 = 0
    num_sym_dim1 = 0
    for dim0, dim1 in zip(list(shape0), list(shape1)):
        if not isinstance(dim0, tir.IntImm):
            num_sym_dim0 += 1
        if not isinstance(dim1, tir.IntImm):
            num_sym_dim1 += 1

    if num_sym_dim0 < num_sym_dim1:
        return shape0
    return shape1


@mutator
class DataflowVarRenamer(PyExprMutator):
    """Rename dataflow var from 'gv' to 'lv'."""

    def __init__(self, mod=None):
        super().__init__(mod)
        self.var2new_var = {}

    def visit_dataflow_var_def_(self, var):
        if var.name_hint.startswith("gv"):
            new_var_name = "lv" + var.name_hint.split("gv")[-1]
            new_var = relax.DataflowVar(new_var_name, var.struct_info, var.span)
            self.var2new_var[var] = new_var
            return new_var
        return var

    def visit_dataflow_var_(self, var):
        return self.var2new_var.get(var, var)


@mutator
class RewriteSubgraph(PyExprMutator):
    """
    A helper class to rewrite expr in while loop function to variable.

    Parameters
    ----------
    rewrite_map : Dict[expr, expr]
        A dictionary contains a set of expr to var mapping.
    """

    def __init__(self, rewrite_map, mod=None):
        super().__init__(mod)
        # check rewrite keys, need to add visit func if missing type.
        for key in rewrite_map.keys():
            assert isinstance(key, (relax.Constant,)), f"Need add visit {type(key)} func."
        self.rewrite_map = rewrite_map

    def visit_constant_(self, const):
        return self.rewrite_map.get(const, const)


class Loop:
    """
    A class contains the components that are used to build up a Relax
    recursive function.
    Parameters
    ----------
    mod : tvm.IRModule
        Module for current parsed IR.

    loop_name : str
        Name prefix of while loop in TensorFlow graph.

    lvar2expr : dict from str to dict from Relax.expr.Var to Relax.expr
        A dictionary recording all loop vars and corresponding
        relax expression.
    """

    def __init__(self, bb, loop_name, lvar2expr):  # pylint: disable=invalid-name
        self.cond = None
        self.body = []
        self._loop = None
        self.bb = bb  # pylint: disable=invalid-name
        self._loop_name = loop_name
        self._lvar2expr = lvar2expr
        self.loop_vars = []

        self.aligned = False

    def _while_loop(self):
        """An internal API to create a Relax recursive function for a matched TF
        `while_loop` construct.
        """
        bind_map = {}
        loop_name = "while_loop_" + self._loop_name.replace("/", "_")
        lv_list = []
        expr_list = []
        extra_vars = []

        for i, lvar in enumerate(self.loop_vars):
            if self._loop_name not in self._lvar2expr:
                self._lvar2expr[self._loop_name] = {}

            # Handle the case when loop var is not properly lifted.
            # This can happen when loop var node name is set accidentally
            # beginning with loop name.
            if lvar not in self._lvar2expr[self._loop_name]:
                var_name = f"{self._loop_name}_loop_var_{i}"
                loop_var = relax.Var(var_name, lvar.struct_info)
                self._lvar2expr[self._loop_name][loop_var] = lvar
                bind_map[lvar] = loop_var
                self.loop_vars[i] = loop_var
                lvar = loop_var

            lv_list.append(lvar)
            expr_list.append(self._lvar2expr[self._loop_name][lvar])

        def _find_bindings(exprs):
            # Find all bindings correspoding to expr and save free vars.
            exprs = exprs if isinstance(exprs, (list, tuple)) else [exprs]
            bindings = []
            visited = set()
            stack = exprs.copy()
            while stack:
                expr = stack.pop()
                if expr in visited:
                    continue
                visited.add(expr)
                if isinstance(expr, relax.Call):
                    stack += list(expr.args)
                elif isinstance(expr, relax.TupleGetItem):
                    stack.append(expr.tuple_value)
                elif isinstance(expr, (relax.Constant, relax.ShapeExpr)):
                    continue
                elif isinstance(expr, relax.Tuple):
                    stack += list(expr.fields)
                elif isinstance(expr, relax.Var):
                    value = self.bb.lookup_binding(expr)
                    if value:
                        bind = relax.VarBinding(expr, value)
                        bindings.insert(0, bind)
                        stack.append(value)
                    elif expr in self._lvar2expr[self._loop_name] and expr not in self.loop_vars:
                        # free vars
                        lv_list.append(expr)
                        expr_list.append(self._lvar2expr[self._loop_name][expr])
                        extra_vars.append(expr)
                else:
                    raise RuntimeError(f"Need to support: {expr}, {type(expr)}")
            return bindings

        def _sort_bindings(bindings):
            # TODO(compass-team): Better method is to do topo logical sort.
            return sorted(bindings, key=lambda x: int(x.var.name_hint[2:]))

        cond_bindings = _find_bindings(self.cond)
        body_bindings = _find_bindings(self.body)

        new_bb = relax.BlockBuilder()
        ret_sinfo = relax.TupleStructInfo([x.struct_info for x in lv_list])
        empty_func = relax.Function.create_empty(lv_list, ret_sinfo)
        new_bb.add_func(empty_func, loop_name)
        loop = self.bb.add_func(empty_func, loop_name)

        # Using bb loop gvar in new_bb loop function.
        cond = self.cond
        if len(cond_bindings) > 0:
            cond_block = relax.BindingBlock(_sort_bindings(cond_bindings))
            cond = new_bb.normalize(relax.SeqExpr([cond_block], cond))

        true_branch = loop(*list(self.body + extra_vars))  # pylint: disable=not-callable
        if len(body_bindings) > 0:
            tb_block = relax.BindingBlock(_sort_bindings(body_bindings))
            true_branch = new_bb.normalize(relax.SeqExpr([tb_block], true_branch))

        false_branch = relax.Tuple(lv_list)
        gvar = relax.If(cond, true_branch, false_branch)
        seqe = new_bb.normalize(relax.SeqExpr([], gvar))
        func = relax.Function(lv_list, seqe)
        if bind_map:
            func = RewriteSubgraph(bind_map).visit_expr(func)
        self.bb.update_func(loop, func)
        loop_ret = loop(*expr_list)
        return loop_ret

    def while_loop(self):
        """Instantiate a while loop if it has not been created yet."""
        if self._loop is None:
            self._loop = self._while_loop()
            return self._loop
        return self._loop


class Branch:
    """A class contains the components that are used to build up a Relax if
    node.

    Parameters
    ----------
    cond : tvm.relax.Expr
        The condition of a if node.

    true_branch : tvm.relax.Expr
        The body of the true branch of a if expression.

    false_branch: tvm.relax.Expr
        The body of the false branch of a if expression.

    _if : tvm.relax.Expr
        An internal variable indicates where an if expression is already created
        for a matched TF condition construct.
    """

    def __init__(self):
        self._if = None
        self.cond = None
        self.true_branch = None
        self.false_branch = None

    def _if_node(self):
        """An internal API to create a relax if node from the matched TF
        condition construct.
        """
        # `cond`  returns a tensor that contains boolean values. We add a `min`
        # operator to checks if there is any false value. If so, this condition
        # doesn't not hold.
        cond = relax.op.min(self.cond)
        return relax.If(cond, self.true_branch, self.false_branch)

    def if_node(self):
        """Create an relax.If node if it hasn't been created yet."""
        if self._if is None:
            self._if = self._if_node()
        return self._if


class TFGraphImporter(object):
    """A helper class for handling Relax graph copying from Tensorflow GraphDef.
    Definition:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto
    """

    def __init__(self, graph, keep_params_in_input):
        self._graph = graph
        self._keep_params_in_input = keep_params_in_input
        self._nodes = {}
        self._tf_node_map = {}
        self._inputs = {}
        self._params = {}
        self._output_shapes = {}
        self._loops = {}
        self._control_flow_node_map = defaultdict(set)
        self._loop_body_order = {}
        self._loop_var_order = {}
        self._lvar2expr = {}
        self._lname_map = {}
        self._sorted_cf_node_names = []
        self._while_loop_name_set = set()
        self._tensor_array_shapes = {}
        self._tensor_array_shape_nodes = {}
        self._branches = {}
        self.bb = relax.BlockBuilder()  # pylint: disable=invalid-name

    def _check_for_unsupported_ops(self):
        missing_operators = set()
        for node in self._graph.node:
            if not any(
                node.op in t
                for t in [
                    ["Placeholder", "Const"],
                    _convert_map,
                    _control_flow_nodes,
                ]
            ):
                missing_operators.add(node.op)

        if missing_operators:
            raise NotImplementedError(
                f"The following operators are not implemented: {missing_operators}"
            )

    def _parse_graph_node(self, shape_dict):
        # pylint: disable=invalid-name, import-outside-toplevel
        try:
            from tensorflow.python.framework import tensor_util
        except ImportError as e:
            raise ImportError(f"Unable to import tensorflow which is required {e}")

        control_flow_nodes = []
        ta_write_nodes = []
        ta_gather_nodes = []
        ta_construct_nodes = []
        for node in self._graph.node:
            node_name_prefix = node.name.rsplit("/", 1)[0]
            self._control_flow_node_map[node_name_prefix].add(node.op)
            self._tf_node_map[node.name] = node

            # Parse output_shapes attribute for each node
            parsed_attr = self._parse_attr(node.attr)
            if "_output_shapes" in parsed_attr:
                self._output_shapes[node.name] = [
                    tensor_util.TensorShapeProtoToList(tshape)
                    for tshape in parsed_attr["_output_shapes"]
                ]
            else:
                self._output_shapes[node.name] = [None]

            if node.op == "Placeholder":
                if shape_dict and node.name in shape_dict:
                    in_shape = list(shape_dict[node.name])
                else:
                    in_shape = tensor_util.TensorShapeProtoToList(node.attr["shape"].shape)
                self._output_shapes[node.name] = [in_shape]
                in_dtype = parsed_attr["dtype"].name
                self._nodes[node.name] = relax.Var(
                    node.name, relax.TensorStructInfo(in_shape, in_dtype)
                )
                self._inputs[node.name] = self._nodes[node.name]
            elif node.op == "Const":
                value = node.attr["value"].tensor
                const_shape = value.tensor_shape
                self._output_shapes[node.name] = [tensor_util.TensorShapeProtoToList(const_shape)]
                if shape_dict and node.name in shape_dict:
                    warnings.warn(
                        f"Ignore the passed shape. Shape in graphdef "
                        f"will be used for operator {node.name}."
                    )

                np_array = tensor_util.MakeNdarray(value)
                const_or_var = self._new_const_or_var(node.name, np_array, shape_dict)
                self._nodes[node.name] = const_or_var
            elif node.op in _control_flow_nodes:
                # We assume that the direct parent node of Exit is a while loop block
                if node.op == "Exit":
                    self._while_loop_name_set.add(node_name_prefix)
                control_flow_nodes.append(node)
            elif node.op.startswith("TensorArray"):
                if is_tensor_array_constuctor(node):
                    ta_construct_nodes.append(node)
                else:
                    for ta_write_name, idx in _tensor_array_write_ops.items():
                        if node.op.startswith(ta_write_name):
                            ta_write_nodes.append((node, idx))
                            break
                    if node.op.startswith("TensorArrayGather"):
                        ta_gather_nodes.append(node)

        # Use tensor array gather to infer static tensor array shape
        for gather_node in ta_gather_nodes:
            input_ta_name = gather_node.input[0]
            input_ta_node = self._tf_node_map[input_ta_name]
            if is_tensor_array_constuctor(input_ta_node):
                gather_attr = self._parse_attr(gather_node.attr)
                if "element_shape" not in gather_attr:
                    continue
                raw_elem_shape = tensor_util.TensorShapeProtoToList(gather_attr["element_shape"])
                elem_shape = []
                dyn_idx = 0
                for dim in raw_elem_shape:
                    if dim < 0:
                        elem_shape.append(tir.Var("m" + str(dyn_idx), "int32"))
                        dyn_idx += 1
                    else:
                        elem_shape.append(int(dim))
                self._tensor_array_shapes[input_ta_node.name] = elem_shape

        # Fetch node contains static tensor array shape
        for item in ta_write_nodes:
            wnode = item[0]
            ta_idx, inode_idx = item[1]

            stack = [self._tf_node_map[wnode.input[ta_idx].split(":")[0]]]
            while stack:
                cnode = stack.pop(0)
                if not cnode.op.startswith("TensorArray"):
                    for iname in cnode.input:
                        stack.append(self._tf_node_map[iname.split(":")[0]])
                elif cnode.name != wnode.name:
                    if is_tensor_array_constuctor(cnode):
                        inode = self._tf_node_map[wnode.input[inode_idx].split(":")[0]]
                        tn = wnode.input[inode_idx].split(":")
                        output_index = int(tn[1]) if len(tn) > 1 else 0
                        self._tensor_array_shape_nodes[cnode.name] = (inode, wnode.op, output_index)
                    break

        # First, parse all control flow nodes.
        # Convert tf.cond to Branch and tf.while_loop to Loop.
        sorted_cf_nodes = []
        exit_pos_map = {}
        ordered_prefix = []
        # Sort control flow nodes to move all Exit nodes to the end
        # of corresponding while_loop block.
        for node in control_flow_nodes:
            loop_name = find_parent_loop_name(node.name, self._while_loop_name_set)
            if node.op == "Exit":
                if loop_name not in exit_pos_map:
                    ordered_prefix.append(loop_name)
                    exit_pos_map[loop_name] = len(sorted_cf_nodes)
                sorted_cf_nodes.append(node)
            elif loop_name in self._while_loop_name_set:
                if loop_name not in exit_pos_map:
                    sorted_cf_nodes.append(node)
                else:
                    sorted_cf_nodes.insert(exit_pos_map[loop_name], node)
                    for j in range(ordered_prefix.index(loop_name), len(ordered_prefix)):
                        exit_pos_map[ordered_prefix[j]] += 1
            else:
                sorted_cf_nodes.append(node)

        for node in sorted_cf_nodes:
            self._sorted_cf_node_names.append(node.name)

        for node in sorted_cf_nodes:
            self._backtrack_construct(node.name)

    def _convert_control_flow_operator(self, node):
        """
        Convert the control flow primitive into corresponding component
        of a Relax control flow construct, i.e. `tf.cond` and `tf.while_loop`
        are converted in Relax `If` and recusrive call, respectively.

        Parameters
        ----------
        node: TensorFlow graph node object.
            A TensorFlow graph node object.

        Returns
        -------
        op : tvm.relax.Expr
            Converted relax expression.
        """
        node_name_prefix = node.name.rsplit("/", 1)[0]
        plname = find_parent_loop_name(node.name, self._while_loop_name_set)
        if node.op == "Merge":
            if _in_while_loop(self._control_flow_node_map, node_name_prefix):
                op = self._licm_construct(plname, node.input[0])
                if node_name_prefix not in self._loops:
                    self._loops[node_name_prefix] = Loop(self.bb, plname, self._lvar2expr)
            else:
                if node_name_prefix not in self._branches:
                    switch_prefix = node_name_prefix + "/Switch"
                    merge_idx = self._sorted_cf_node_names.index(node.name)
                    for i in range(merge_idx - 1, -1, -1):
                        cf_name = self._sorted_cf_node_names[i]
                        if cf_name.startswith(switch_prefix):
                            self._backtrack_construct(cf_name)
                            break
                if node_name_prefix in self._branches:
                    branch = self._branches[node_name_prefix]
                else:
                    for cur_br in self._branches:
                        if cur_br.startswith(node_name_prefix):
                            branch = self._branches[cur_br]
                            break

                false_br = self._licm_construct(plname, node.input[0])
                true_br = self._licm_construct(plname, node.input[1])
                branch.true_branch = true_br
                branch.false_branch = false_br
                if node_name_prefix not in self._while_loop_name_set:
                    assert isinstance(branch.cond, relax.Constant), "Unsupport now."
                    assert branch.cond.data.numpy() in [True, False]
                    if branch.cond.data.numpy():
                        op = branch.true_branch
                    else:
                        op = branch.false_branch
        elif node.op == "Exit":
            loop = self._loops[node_name_prefix]

            # Check whether the order of loop variables aligns
            # with loop body. If not, create new loop variable list
            # with correct order.
            if not loop.aligned:
                loop_vars = []
                for i in self._loop_body_order[node_name_prefix]:
                    for j, k in enumerate(self._loop_var_order[node_name_prefix]):
                        if k == i:
                            loop_vars.append(loop.loop_vars[j])
                loop.loop_vars = loop_vars
                loop.aligned = True
            exit_name = node.name.split("/")[-1]
            if "_" in exit_name:
                exit_number = int(exit_name[5:])
            else:
                exit_number = 0
            expr = loop.while_loop()
            body_pos = exit_number
            for i, j in enumerate(self._loop_body_order[node_name_prefix]):
                if exit_number == j:
                    body_pos = i
                    break
            op = self.bb.normalize(expr[body_pos])
        elif node.op == "Enter":
            op = self._licm_construct(plname, node.input[0])
        elif node.op == "LoopCond":
            op = self._licm_construct(plname, node.input[0])
            self._loops[node_name_prefix].cond = op
        elif node.op == "Switch":
            op = self._licm_construct(plname, node.input[0])
            cond = self._licm_construct(plname, node.input[1])
            if _in_while_loop(self._control_flow_node_map, node_name_prefix):
                if node_name_prefix not in self._loop_var_order:
                    self._loop_var_order[node_name_prefix] = []
                if node.name.endswith("Switch"):
                    self._loop_var_order[node_name_prefix].append(0)
                else:
                    var_idx = int(node.name.split("Switch_")[-1])
                    self._loop_var_order[node_name_prefix].append(var_idx)
                self._loops[node_name_prefix].loop_vars.append(op)
            else:
                if node_name_prefix not in self._branches:
                    self._branches[node_name_prefix] = Branch()
                self._branches[node_name_prefix].cond = cond
        elif node.op == "NextIteration":
            if node_name_prefix not in self._loop_body_order:
                self._loop_body_order[node_name_prefix] = []
            if node.name.endswith("NextIteration"):
                self._loop_body_order[node_name_prefix].append(0)
            else:
                self._loop_body_order[node_name_prefix].append(
                    int(node.name.split("NextIteration_")[-1])
                )
            op = self._licm_construct(plname, node.input[0])
            self._loops[node_name_prefix].body.append(op)
        else:
            raise Exception(f"Cannot identify control flow operator: {node.op}")

        return op

    def _licm_construct(self, loop_name, node_name):
        """Construct a node by considering whether it is
        loop invariant with the given while loop. If yes, we
        generate a loop Variable. Otherwise, return regular
        converted relax expression.

        Parameters
        ----------
        loop_name : str
            TensorFlow while loop name to be checked.

        node_name : str
            TensorFlow node name.

        Returns
        -------
        out : relax.Expr or relax.Var
            Converted relax expression or loop var.
        """
        actual_expr = self._backtrack_construct(node_name)
        tname = node_name.split(":")
        node_name = tname[0].split("^")[-1]
        cloop_name = find_parent_loop_name(node_name, self._while_loop_name_set)

        if loop_name in self._while_loop_name_set and not cloop_name.startswith(loop_name):
            if loop_name not in self._lvar2expr:
                self._lvar2expr[loop_name] = {}
            if loop_name not in self._lname_map:
                self._lname_map[loop_name] = {}

            if node_name not in self._lname_map[loop_name]:
                var_name = f"{node_name}_loop_var"
                var_sinfo = actual_expr.struct_info
                loop_var = tvm.relax.Var(var_name, var_sinfo)
                # try:
                #     extra_param = _infer_value(actual_expr, self._params, self._mod)
                #     self._params[var_name] = extra_param
                # except Exception:
                #     pass
                self._lvar2expr[loop_name][loop_var] = actual_expr
                self._lname_map[loop_name][node_name] = loop_var
                ret = loop_var
            else:
                ret = self._lname_map[loop_name][node_name]
        else:
            ret = actual_expr

        return ret

    def _backtrack_construct(self, node_name):
        """Convert a specific tensorflow node to relax expression.

        If any of its ancestor node is not converted yet, backtrack as
        far as input node and covert all nodes on the path.

        This is required when parsing control flow nodes, since the parsing
        order may not follow the original graph def.

        Parameters
        ----------
        node_name : str
            TensorFlow node name.

        Returns
        -------
        op : relax.Expr
            Converted relax expression
        """
        # pylint: disable=invalid-name, import-outside-toplevel
        try:
            from tensorflow.python.framework import tensor_util
        except ImportError as e:
            raise ImportError(f"Unable to import tensorflow which is required {e}")

        input_op_name = node_name.split(":")[0].split("^")[-1]
        if input_op_name not in self._nodes:
            node = self._tf_node_map[input_op_name]
            attr = self._parse_attr(node.attr)

            if node.op in _control_flow_nodes:
                attr = self._parse_attr(node.attr)
                op = self._convert_control_flow_operator(node)
            else:
                attr["_output_shapes"] = self._output_shapes[input_op_name]
                attr["_node_name"] = node.name

                inputs = [self._backtrack_construct(iname) for iname in node.input]

                plname = find_parent_loop_name(node_name, self._while_loop_name_set)

                # For TensorArrayV3 op, we need to infer shape first
                if is_tensor_array_constuctor(node):
                    raw_elem_shape = tensor_util.TensorShapeProtoToList(attr["element_shape"])
                    elem_shape = []
                    dyn_idx = 0
                    for dim in raw_elem_shape:
                        if dim < 0:
                            elem_shape.append(tir.Var("m" + str(dyn_idx), "int32"))
                            dyn_idx += 1
                        else:
                            elem_shape.append(dim)

                    if elem_shape:
                        attr["shape"] = elem_shape

                    # it should infer shape no matter whether elem_shape.
                    shape_node, wnode_op, output_index = self._tensor_array_shape_nodes[node.name]
                    name = shape_node.name
                    if output_index > 0:
                        name += ":" + str(output_index)
                    converted = self._backtrack_construct(name)
                    shape = converted.struct_info.shape.values
                    if wnode_op.startswith("TensorArraySplit"):
                        shape = (tir.Var("m", "int32"),) + shape[1:]
                    elif wnode_op.startswith("TensorArrayScatter"):
                        shape = shape[1:]

                    if node.name in self._tensor_array_shapes:
                        preset_shape = self._tensor_array_shapes[node.name]
                        shape = _get_more_static_shape(shape, preset_shape)

                    if "shape" in attr:
                        attr["shape"] = _get_more_static_shape(shape, attr["shape"])
                    else:
                        attr["shape"] = shape

                # LICM
                if plname in self._while_loop_name_set:
                    for i, iname in enumerate(node.input):
                        actual_input = self._licm_construct(plname, iname)
                        inputs[i] = actual_input

                op = self._convert_operator(node.op, inputs, attr)

            if not isinstance(op, relax.Tuple):
                if isinstance(op.checked_type, tvm.ir.type.TupleType):
                    # This is a var bound to a tuple. We need to unpack it and create
                    # a new tuple.
                    tuple_items = []
                    for i in range(len(op.checked_type.fields)):
                        tuple_items.append(self.bb.emit(relax.TupleGetItem(op, i)))
                    op = relax.Tuple(tuple_items)

            self._nodes[input_op_name] = op

        out = self._nodes[input_op_name]
        if isinstance(out, relax.Tuple):
            splited = node_name.split(":")
            tensor_slot = int(splited[1]) if len(splited) > 1 else 0
            return out[tensor_slot]

        return out

    def _construct_nodes(self):
        for node in self._graph.node:
            self._backtrack_construct(node.name)

    def from_tensorflow(self, shape_dict=None, outputs=None):
        """Construct Relax expressions from the tensorflow graph.

        Parameters
        ----------
        shape_dict : Dictionary of input dimensions (Optional)
            Graph level input shape dictionary.

        outputs : List of output tensor names (Optional)
        if not specified then the last node is assumed as graph output.

        Returns
        -------
        mod : tvm.IRModule
            The returned relax module
        """
        with self.bb.function("main"):
            self._check_for_unsupported_ops()
            self._parse_graph_node(shape_dict)
            self._construct_nodes()

            if outputs is None:
                out = self._nodes[self._graph.node[-1].name.split(":")[0]]
            else:
                out = [self._nodes[x.split(":")[0]] for x in outputs]
                out = out[0] if len(out) == 1 else relax.Tuple(out)
            out = self.bb.emit(out)

            # Create function attributes for this module
            func_attrs = {"num_input": len(self._inputs)}
            # Create a function from our output expression and all input variables.
            input_list = [value for value in self._inputs.values() if isinstance(value, relax.Var)]
            # Attach params if they are available.
            if self._keep_params_in_input and self._params:
                param_var_list, param_value_list = map(list, zip(*self._params.items()))
                input_list += param_var_list
                func_attrs["params"] = param_value_list

            self.bb.emit_func_output(out, params=input_list)

        relax_mod = self.bb.get()
        relax_mod = relax.transform.RemoveUnusedOutputs()(relax_mod)
        relax_mod = relax.transform.ConvertToDataflow(1)(relax_mod)
        relax_mod["main"] = DataflowVarRenamer().visit_expr(relax_mod["main"])
        # Attach attributes.
        relax_mod["main"] = relax_mod["main"].with_attrs(func_attrs)
        return relax_mod

    def _new_const_or_var(self, node_name, value, shape=None, dtype=None):
        dtype = value.dtype if dtype is None else dtype
        if dtype == np.dtype(object):
            # Object types are generally tensorflow DT_STRING (DecodeJpeg op).
            # Just leave it as placeholder.
            if shape and node_name in shape:
                var_shape = shape[node_name]
            else:
                var_shape = self._output_shapes[node_name][0]
            return relax.Var(node_name, relax.TensorStructInfo(var_shape, "uint8"))
        if self._keep_params_in_input:
            new_var = relax.Var(node_name, relax.TensorStructInfo(value.shape, dtype))
            self._params[new_var] = tvm.nd.array(value)
            return new_var

        return relax.const(value, dtype)

    def _get_attr(self, buf):
        """Returns the value of the attr of this buf with the given `name`.

        Args:
          buf: attrvalue protobuf.

        Returns:
          The value of the attr, as a Python object.

        Raises:
          ValueError: If this op does not have an attr with the given `name`.
        """
        # pylint: disable=invalid-name, import-outside-toplevel
        try:
            from tensorflow.python.framework import dtypes
        except ImportError as e:
            raise ImportError(f"Unable to import tensorflow which is required {e}")

        fields = ["s", "i", "f", "b", "type", "shape", "tensor", "func"]
        x = buf
        ret = []

        # Treat an empty oneof value as an empty list.
        if not x.WhichOneof("value"):
            return ret
        if x.HasField("list"):
            for f in fields:
                if getattr(x.list, f):
                    if f == "type":
                        ret += [dtypes.as_dtype(x) for x in list(getattr(x.list, f))]
                    else:
                        ret += list(getattr(x.list, f))
        else:
            for f in fields:
                if x.HasField(f):
                    if f == "type":
                        ret = dtypes.as_dtype(getattr(x, f))
                    else:
                        ret = getattr(x, f)
        return ret

    def _parse_attr(self, attr_proto):
        """Convert a list of AttributeProto to a dict, with names as keys."""
        attrs = {}
        for key, value in attr_proto.items():
            attrs[key] = self._get_attr(value)

        return attrs

    def _convert_operator(self, op_name, inputs, attrs):
        # pylint: disable=import-outside-toplevel
        from tvm.compass.relax.op import TF_CUSTOM_OP_DICT

        convert_map = _convert_map
        if op_name in TF_CUSTOM_OP_DICT:
            converter = TF_CUSTOM_OP_DICT[op_name]
            op = converter(inputs, attrs, self._params, self.bb)
        elif op_name in convert_map:
            op = convert_map[op_name](inputs, attrs, self._params, self.bb)
        else:
            raise NotImplementedError(f"Operator {op_name} not implemented.")

        if isinstance(op, np.ndarray):
            op = self._new_const_or_var(attrs["_node_name"], op)
        op = self.bb.normalize(op)

        return op


def from_tensorflow(graph, shape_dict=None, keep_params_in_input=False, outputs=None):
    """Load tensorflow graph which is a python tensorflow graph object into relax.
    The companion parameters will be handled automatically.

    Parameters
    ----------
    graph : GraphDef object
        Tensorflow GraphDef

    shape_dict : Dictionary of input dimensions (Optional)
        Graph level input shape dictionary.

    keep_params_in_input : bool
        If True, parameters will be treated as input variables. If false,
        parameters are treated as constant and folded directly into the graph.

    outputs : List of output tensor names (Optional)
        if not specified then the last node is assumed as graph output.

    Returns
    -------
    mod : tvm.IRModule
        The relax module for compilation.
    """
    g = TFGraphImporter(graph, keep_params_in_input)
    return g.from_tensorflow(shape_dict, outputs)
