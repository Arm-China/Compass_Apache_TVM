# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
"""TF: Tensorflow frontend."""
import warnings
import numpy as np
import tvm
from tvm import relax
from .tensorflow_ops import _convert_map


__all__ = ["from_tensorflow"]


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
        self.bb = relax.BlockBuilder()  # pylint: disable=invalid-name

    def _check_for_unsupported_ops(self):
        missing_operators = set()
        for node in self._graph.node:
            if not any([node.op in t for t in [["Placeholder", "Const"], _convert_map]]):
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

        for node in self._graph.node:
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
                const_or_var = self._new_const_or_var(node.name, np_array)
                self._nodes[node.name] = const_or_var

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
        input_op_name = node_name.split(":")[0].split("^")[-1]
        if input_op_name not in self._nodes:
            node = self._tf_node_map[input_op_name]
            parsed_attr = self._parse_attr(node.attr)

            parsed_attr["_output_shapes"] = self._output_shapes[input_op_name]
            parsed_attr["_node_name"] = node.name

            inputs = [self._backtrack_construct(iname) for iname in node.input]

            op = self._convert_operator(node.op, inputs, parsed_attr)

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
            with self.bb.dataflow():
                self._check_for_unsupported_ops()
                self._parse_graph_node(shape_dict)
                self._construct_nodes()

                if outputs is None:
                    out = self._nodes[self._graph.node[-1].name.split(":")[0]]
                else:
                    out = [self._nodes[x.split(":")[0]] for x in outputs]
                    out = out[0] if len(out) == 1 else relax.Tuple(out)

                output_var = self.bb.emit_output(out)

            # Create function attributes for this module
            func_attrs = {"num_input": len(self._inputs)}
            # Create a function from our output expression and all input variables.
            input_list = [value for value in self._inputs.values() if isinstance(value, relax.Var)]
            # Attach params if they are available.
            if self._keep_params_in_input and self._params:
                param_var_list, param_value_list = map(list, zip(*self._params.items()))
                input_list += param_var_list
                func_attrs["params"] = param_value_list

            self.bb.emit_func_output(output_var, params=input_list)

        relax_mod = self.bb.get()
        # Attach attributes.
        relax_mod["main"] = relax_mod["main"].with_attrs(func_attrs)
        return relax_mod

    def _new_const_or_var(self, node_name, value, dtype=None):
        dtype = value.dtype if dtype is None else dtype
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
