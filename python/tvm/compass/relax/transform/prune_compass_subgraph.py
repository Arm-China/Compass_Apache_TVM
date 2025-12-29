# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
"""Prune computatin non-intensive Compass subgraphs."""
import math
from tvm import relax, ir, DataType
from tvm.compass.logger import DEBUG
from tvm.relax.expr_functor import PyExprVisitor, visitor
from tvm.compass.relax.config import CompassConfig
from .utils import is_compass_func


@visitor
class IsComputeIntensiveGraph(PyExprVisitor):
    """Check if is compute-intensive subgraph."""

    def __init__(self):
        self.total_weighted_compute_count = 0
        self.compute_count_threshold = float(CompassConfig.get().common["compute_threshold"])

    def get_memory_access_count(self, call):
        """Calculate memory access count of call node.
        count = count-input + count-weight + count-output
        """
        count = 0
        sinfos = []
        for arg in call.args:
            sinfo = arg.struct_info
            if isinstance(sinfo, relax.TupleStructInfo):
                for t in sinfo.fields:
                    sinfos.append(t)
            else:
                sinfos.append(sinfo)

        out_sinfo = call.struct_info
        if isinstance(out_sinfo, relax.TupleStructInfo):
            for field in out_sinfo.fields:
                sinfos.append(field)
        else:
            sinfos.append(out_sinfo)

        for sinfo in sinfos:
            if not hasattr(sinfo, "shape"):
                continue
            shape = [int(x) for x in sinfo.shape]
            dtype = DataType(sinfo.dtype)
            count += math.prod(shape) * dtype.bits / 8

        return count

    def visit_call_(self, call):
        if self.total_weighted_compute_count >= self.compute_count_threshold:
            return
        if not isinstance(call.op, ir.Op):
            return
        # Operation compute count. means FLOPs if dtype is float.
        compute_count = 0
        if call.op.has_attr("FComputeCount"):
            f_cc = call.op.get_attr("FComputeCount")
            compute_count = f_cc(call) if f_cc else compute_count

        # Memory access count (Bytes)
        if call.op.has_attr("FMemoryAccessCount") and call.op.get_attr("FMemoryAccessCount"):
            f_memory_access_count = call.op.get_attr("FMemoryAccessCount")
            memory_access_count = f_memory_access_count(call)
        else:
            memory_access_count = self.get_memory_access_count(call)

        # Intensity
        node_intensity = compute_count / memory_access_count

        weighted_compute_count = compute_count * node_intensity
        self.total_weighted_compute_count += weighted_compute_count
        return

    def is_intensive(self, func):
        """
        Visits the graph and checks if it's compute intensive."
        """
        self.total_weighted_compute_count = 0
        self.visit_expr(func)
        return self.total_weighted_compute_count >= self.compute_count_threshold


@ir.transform.module_pass(opt_level=0)
class PruneCompassSubGraphs:
    """Prune computatin non-intensive Compass subgraphs."""

    def transform_module(self, ir_mod, pass_ctx):  # pylint: disable=unused-argument
        """Traverse the given IRModule, transform it and return the result."""
        all_small_subgraph = True
        global_vars_to_inline = []
        debug_msg = {}
        checker = IsComputeIntensiveGraph()
        # Collect all gvar to be inlined.
        for gvar, func in ir_mod.functions.items():
            if not is_compass_func(func):
                continue
            if not checker.is_intensive(func):
                debug_msg[gvar] = checker.total_weighted_compute_count
                global_vars_to_inline.append(gvar)
            else:
                all_small_subgraph = False

        if all_small_subgraph:
            return ir_mod

        # Dump inlined subfunc message.
        for k, v in debug_msg.items():
            DEBUG(f"{k.name_hint} will be inlined to cpu with compute count:{v}")

        # Inline composite inner func in subfunc.
        for gvar in global_vars_to_inline:
            name = gvar.name_hint
            func = ir_mod[gvar]
            tmp_mod = ir.IRModule.from_expr(func.without_attr("Codegen"))
            tmp_mod = relax.transform.LambdaLift()(tmp_mod)
            for inner_gvar, inner_func in tmp_mod.functions.items():
                if inner_gvar.name_hint == name:
                    continue
                tmp_mod[name] = tmp_mod[name].inline_functions({inner_gvar.name_hint: inner_func})
            tmp_mod = relax.transform.ConvertToDataflow(1)(tmp_mod)
            ir_mod[gvar] = tmp_mod[name]

        # Inline subfunc in each caller.
        for gvar, func in ir_mod.functions.items():
            if gvar in global_vars_to_inline:
                continue
            candidates_in_cur_func = []
            for block in func.body.blocks:
                for bind in block.bindings:
                    if not isinstance(bind.value, relax.Call):
                        continue
                    if bind.value.op in global_vars_to_inline:
                        candidates_in_cur_func.append(bind.value.op)
            for candidate in candidates_in_cur_func:
                func = func.inline_functions({candidate.name_hint: ir_mod[candidate]})
            ir_mod[gvar] = func

        # Delete all inlined subfunc.
        for gvar in global_vars_to_inline:
            del ir_mod[gvar]
        return ir_mod
