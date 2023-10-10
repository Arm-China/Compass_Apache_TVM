# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
"""Extract the head quantize and tail dequantize nodes of AIPU subgraph."""
from tvm import relay, ir


class InOutQuantOpExtractor(relay.ExprMutator):
    """Extract the head quantize and tail dequantize nodes of AIPU subgraph."""

    def __init__(self):
        super().__init__()
        self._quantize_ops = []
        self._dequantize_ops = []
        self._params = []
        self._outputs = []

    def visit_call(self, call):
        if call.op == relay.op.get("qnn.quantize"):
            data = call.args[0]
            if data in self._params:
                idx = self._params.index(data)
                self._quantize_ops[idx] = call
                new_var = relay.var(data.name_hint, call.checked_type)
                self._params[idx] = new_var
                return new_var

        if call.op == relay.op.get("qnn.dequantize"):
            if call in self._outputs:
                self._dequantize_ops[self._outputs.index(call)] = call
                return self.visit(call.args[0])

        return super().visit_call(call)

    def extract(self, func):
        """Extract the head quantize and tail dequantize nodes from the given function."""
        self._params = list(func.params)
        self._quantize_ops = [None] * len(self._params)

        body = func.body
        self._outputs = list(body.fields) if isinstance(body, relay.Tuple) else [body]
        self._dequantize_ops = [None] * len(self._outputs)

        # Collect the head quantize and tail dequantize nodes and remove them from the function.
        new_body = self.visit(body)
        new_func = func
        if self._params != list(func.params) or new_body != body:
            new_func = relay.Function(self._params, new_body, attrs=func.attrs)
        return new_func, self._quantize_ops, self._dequantize_ops


class CallerQuantOpInjector(relay.ExprMutator):
    """Inject the extracted quantize relevant nodes to the caller."""

    def __init__(self, extracted_ops):
        super().__init__()
        self._extracted_ops = extracted_ops

    def visit_call(self, call):
        ret = super().visit_call(call)

        if ret.op not in self._extracted_ops:
            return ret

        # This node is a call to a changed AIPU subgraph.
        quantize_ops, dequantize_ops = self._extracted_ops[ret.op]

        # Inject the extracted quantize nodes after the corresponding arguments.
        assert len(quantize_ops) == len(ret.args)
        new_args = []
        for quant, arg in zip(quantize_ops, ret.args):
            if quant is None:
                # The type of this argument originally is quantized.
                new_args.append(arg)
                continue
            new_args.append(relay.expr.CallWithFields(quant, args=([arg] + quant.args[1:])))

        # Create the new call with the new arguments.
        ret = relay.Call(ret.op, new_args, ret.attrs)

        # Inject the extracted dequantize nodes after the corresponding output nodes.
        if isinstance(call.checked_type, ir.TupleType):
            new_fields = []
            assert len(dequantize_ops) == len(call.checked_type.fields)
            for i, dequant in enumerate(dequantize_ops):
                tgi = relay.TupleGetItem(ret, i)
                if dequant is None:
                    # The type of this output node originally is quantized.
                    new_fields.append(tgi)
                    continue
                new_dequant_args = [tgi] + dequant.args[1:]
                new_fields.append(relay.expr.CallWithFields(dequant, args=new_dequant_args))
            return relay.Tuple(new_fields)

        assert len(dequantize_ops) == 1
        dequant = dequantize_ops[0]
        if dequant is not None:
            return relay.expr.CallWithFields(dequant, args=([ret] + dequant.args[1:]))

        return ret


@ir.transform.module_pass(opt_level=0, required=("InferType",))
class ExtractInOutQuantOps:
    """Extract the head quantize and tail dequantize nodes of AIPU subgraph to the caller."""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        new_ir_mod = ir_mod.shallow_copy()
        # Delete the caller first, so the type inference can work correctly.
        new_ir_mod.remove("main")

        # Processing all AIPU subgraphs.
        extracted_ops = dict()
        extractor = InOutQuantOpExtractor()
        for gvar, func in ir_mod.functions.items():
            if hasattr(func.attrs, "Compiler") and func.attrs.Compiler == "aipu_compass":
                new_func, quantize_ops, dequantize_ops = extractor.extract(func)
                if new_func == func:
                    continue
                extracted_ops[gvar] = (quantize_ops, dequantize_ops)
                new_ir_mod[gvar] = new_func
                # Do type inference once a new function is updated into the IRModule.
                new_ir_mod = relay.transform.InferType()(new_ir_mod)

        if len(extracted_ops) == 0:
            return ir_mod

        # Processing the main function.
        new_ir_mod["main"] = CallerQuantOpInjector(extracted_ops).visit(ir_mod["main"])
        return relay.transform.InferType()(new_ir_mod)
