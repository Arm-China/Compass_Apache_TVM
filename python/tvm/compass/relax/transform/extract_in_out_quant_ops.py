# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Extract the head quantize and tail dequantize nodes of Compass subgraph for Relax."""
from tvm import relax, ir
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.analysis import get_var2val
from .utils import is_compass_func


@mutator
class InOutQuantOpExtractor(PyExprMutator):
    """Extract the head quantize and tail dequantize nodes of Compass subgraph for Relax."""

    def __init__(self):
        super().__init__()
        self._quantize_ops = []
        self._dequantize_ops = []
        self._params = []
        self._outputs = []

    def visit_call_(self, call):
        if isinstance(call.op, ir.Op) and call.op.name == "relax.quantize":
            data = call.args[0]
            if data in self._params:
                idx = self._params.index(data)
                self._quantize_ops[idx] = call
                new_var = relax.Var(data.name_hint, call.struct_info)
                self._params[idx] = new_var
                return new_var

        if isinstance(call.op, ir.Op) and call.op.name == "relax.dequantize":
            if call in self._outputs:
                idx = self._outputs.index(call)
                self._dequantize_ops[idx] = call
                return self.visit_expr(call.args[0])

        return super().visit_call_(call)

    def extract(self, func: relax.Function):
        """Extract the head quantize and tail dequantize nodes from the given Relax function."""
        self._params = list(func.params)
        self._quantize_ops = [None] * len(self._params)

        var2val = get_var2val(func)
        body = func.body
        output_expr = body.body
        output_expr = var2val.get(output_expr, output_expr)
        if isinstance(output_expr, relax.Tuple):
            self._outputs = list(output_expr.fields)
        else:
            self._outputs = [output_expr]
        self._outputs = [var2val.get(var, var) for var in self._outputs]

        self._dequantize_ops = [None] * len(self._outputs)

        # Collect the head quantize and tail dequantize nodes and remove them from the function.
        new_body = self.visit_expr(body)
        new_func = func

        if self._params != list(func.params) or new_body != body:
            new_func = relax.Function(
                self._params, new_body, new_body.struct_info, func.is_pure, func.attrs
            )
        return new_func, self._quantize_ops, self._dequantize_ops


@mutator
class CallerQuantOpInjector(PyExprMutator):
    """Inject the extracted quantize and dequantize nodes to the caller for Relax."""

    def __init__(self, extracted_ops, gv2new_gv):
        super().__init__()
        self._extracted_ops = extracted_ops
        self._gv2new_gv = gv2new_gv

    def visit_call_(self, call):
        ret = super().visit_call_(call)
        if ret.op not in self._extracted_ops:
            return ret

        # This node is a call to a changed Compass subgraph.
        quantize_ops, dequantize_ops = self._extracted_ops[ret.op]

        # Inject the extracted quantize nodes after the corresponding arguments.
        assert len(quantize_ops) == len(ret.args)
        new_args = []
        for quant_op, arg in zip(quantize_ops, ret.args):
            if quant_op is None:
                # The type of this argument originally is quantized.
                new_args.append(arg)
                continue
            new_quant_call = relax.Call(
                quant_op.op, [arg] + list(quant_op.args[1:]), quant_op.attrs, quant_op.sinfo_args
            )
            new_args.append(new_quant_call)

        # Create the new call with the new arguments.
        new_call = relax.Call(self._gv2new_gv[ret.op], new_args, ret.attrs, ret.sinfo_args)
        if dequantize_ops is None or len(dequantize_ops) == 0:
            return new_call

        # Inject the extracted dequantize nodes after the corresponding output nodes.
        if isinstance(ret.struct_info, relax.TupleStructInfo):
            new_fields = []
            assert len(dequantize_ops) == len(ret.struct_info.fields)
            for i, dequant_op in enumerate(dequantize_ops):
                tuple_get_item = relax.TupleGetItem(new_call, i)
                if dequant_op is None:
                    # The type of this output node originally is quantized.
                    new_fields.append(tuple_get_item)
                    continue
                new_dequant_call = relax.Call(
                    dequant_op.op, [tuple_get_item] + list(dequant_op.args[1:]), dequant_op.attrs
                )
                new_fields.append(new_dequant_call)
            return relax.Tuple(new_fields)
        else:
            assert len(dequantize_ops) == 1
            dequant_op = dequantize_ops[0]
            if dequant_op is not None:
                return relax.Call(
                    dequant_op.op, [new_call] + list(dequant_op.args[1:]), dequant_op.attrs
                )
            return new_call


@ir.transform.module_pass(opt_level=0)
class ExtractInOutQuantOps:
    """Extract the head quantize and tail dequantize nodes of Compass subgraph
    to the caller for Relax."""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        extracted_ops = {}
        gv2new_gv = {}

        # Processing all Compass subgraphs.
        extractor = InOutQuantOpExtractor()
        for gvar, func in ir_mod.functions.items():
            if not is_compass_func(func):
                continue

            new_func, quant_ops, dequant_ops = extractor.extract(func)
            if new_func != func:
                new_gv = relax.GlobalVar(gvar.name_hint)
                relax.expr._update_struct_info(new_gv, new_func.struct_info)
                del ir_mod[gvar]
                ir_mod[new_gv] = new_func
                gv2new_gv[gvar] = new_gv
                extracted_ops[gvar] = (quant_ops, dequant_ops)

        if not extracted_ops:
            return ir_mod

        # Processing the main function.
        injector = CallerQuantOpInjector(extracted_ops, gv2new_gv)
        for gvar, func in ir_mod.functions.items():
            if is_compass_func(func):
                continue
            new_func = injector.visit_expr(func)
            if new_func != func:
                ir_mod[gvar] = new_func
        ir_mod = relax.transform.RemoveUnusedOutputs()(ir_mod)

        return ir_mod
