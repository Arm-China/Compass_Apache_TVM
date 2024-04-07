# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
# pylint: disable=bad-super-call
"""Prune computatin non-intensive AIPU subgraphs."""
import numpy as np
import tvm
from tvm import relay, ir
from tvm.aipu.logger import DEBUG
from tvm.relay.backend.contrib.aipu_compass.config import AipuCompassConfig


@ir.register_op_attr("nn.conv2d", "FComputeCount")
@ir.register_op_attr("nn.conv2d_transpose", "FComputeCount")
@ir.register_op_attr("nn.dense", "FComputeCount")
@ir.register_op_attr("nn.batch_matmul", "FComputeCount")
def _count(call: relay.Call):
    mac_count_func = call.op.get_attr("FMacCount")
    return mac_count_func(call) * 2


@ir.register_op_attr("qnn.conv2d", "FComputeCount")
def _count(call: relay.Call):
    attr = call.attrs
    input_shape = call.args[0].checked_type.shape
    output_shape = list(call.checked_type.shape)
    kernel_size = list(attr.kernel_size)
    if attr.data_layout == "NHWC":
        input_channel = input_shape[3]
    elif attr.data_layout == "NCHW":
        input_channel = input_shape[1]
    else:
        raise ValueError("Unsupport data layout in compute count for qnn.conv2d.")
    count = np.prod(np.array(output_shape), dtype="int64")
    count *= np.prod(np.array(kernel_size), dtype="int64")
    assert input_channel % attr.groups == 0
    count *= int(input_channel) / int(attr.groups)
    count *= 2
    return count


@ir.register_op_attr("multiply", "FComputeCount")
def _count(call: relay.Call):
    input_shape = call.args[0].checked_type.shape
    compute_count = 1
    for dim in input_shape:
        compute_count *= int(dim)

    return compute_count


@ir.register_op_attr("contrib.aipu_compass.gruv3", "FComputeCount")
def _count(gruv3: relay.Call):
    bias = gruv3.args[3]
    input_size = int(gruv3.args[0].checked_type.shape[2])
    cell_size = int(bias.checked_type.shape[0]) // 3
    count = 2 * 3 * cell_size * (cell_size + input_size)

    return count


@ir.register_op_attr("multiply", "FMemoryAccessCount")
def _count(call: relay.Call):
    tensor_types = []
    tensor_types.append(call.args[0].checked_type)
    tensor_types.append(call.args[1].checked_type)
    tensor_types.append(call.checked_type)

    count = 0
    for ttype in tensor_types:
        shape = list(ttype.shape)
        dtype = tvm.DataType(ttype.dtype)
        count += np.prod(np.array(shape, dtype="int64")) * dtype.bits / 8

    return count


@ir.register_op_attr("contrib.aipu_compass.gruv3", "FMemoryAccessCount")
def _count(gruv3: relay.Call):
    bias = gruv3.args[3]
    input_size = int(gruv3.args[0].checked_type.shape[2])
    cell_size = int(bias.checked_type.shape[0]) // 3
    dtype = gruv3.args[0].checked_type.dtype
    bits = tvm.DataType(dtype).bits
    count = 3 * cell_size * (cell_size + input_size + 2) * bits / 8

    return count


class IsComputeIntensiveGraph(relay.ExprMutator):
    """Check if is compute-intensive subgraph."""

    def __init__(self, subgraph: relay.expr.Expr):
        super().__init__()
        self.subgraph = subgraph
        self.total_weighted_compute_count = 0
        self.compute_count_threshold = float(AipuCompassConfig.get().common["compute_threshold"])

    def get_memory_access_count(self, call):
        """Calculate memory access count of call node.
        count = count-input + count-weight + count-output
        """
        count = 0
        tensor_types = []
        for arg in call.args:
            ttypes = arg.checked_type
            if isinstance(ttypes, relay.TupleType):
                for t in ttypes.fields:
                    tensor_types.append(t)
            else:
                tensor_types.append(ttypes)

        if isinstance(call.checked_type, relay.TupleType):
            for field in call.checked_type.fields:
                tensor_types.append(field)
        else:
            tensor_types.append(call.checked_type)

        for ttype in tensor_types:
            shape = list(ttype.shape)
            dtype = tvm.DataType(ttype.dtype)
            count += np.prod(np.array(shape, dtype="int64")) * dtype.bits / 8

        return count

    def visit_call(self, call):
        if self.total_weighted_compute_count >= self.compute_count_threshold:
            return super().visit_call(call)

        if isinstance(call.op, relay.Function):
            self.visit(call.op.body)
            return super().visit_call(call)

        # Operation compute count. means FLOPs if dtype is float.
        compute_count = 0
        f_compute_count = call.op.get_attr("FComputeCount")
        if f_compute_count is not None:
            compute_count = f_compute_count(call)

        # Memory access count (Bytes)
        memory_access_count = 0
        f_memory_access_count = call.op.get_attr("FMemoryAccessCount")
        if f_memory_access_count is None:
            memory_access_count = self.get_memory_access_count(call)
        else:
            memory_access_count = f_memory_access_count(call)

        # Intensity
        node_intensity = compute_count / memory_access_count

        weighted_compute_count = compute_count * node_intensity
        self.total_weighted_compute_count += weighted_compute_count

        return super().visit_call(call)

    def check(self) -> bool:
        """
        Visits the graph and checks if it's compute intensive."
        """
        self.visit(self.subgraph)

        return self.total_weighted_compute_count >= self.compute_count_threshold


@tvm.ir.transform.module_pass(opt_level=0)
class PruneAIPUSubGraphs:
    """
    Prune computatin non-intensive AIPU subgraphs.
    """

    def transform_module(self, mod, ctx):  # pylint: disable=unused-argument
        """try transform module"""
        # find functions need to inline.
        all_small_subgraph = True
        global_vars_to_inline = []
        debug_msg = {}
        for gvar in mod.get_global_vars():
            if (
                mod[gvar].attrs
                and "Compiler" in mod[gvar].attrs
                and mod[gvar].attrs["Compiler"] == "aipu_compass"
            ):
                checker = IsComputeIntensiveGraph(mod[gvar].body)
                if not checker.check():
                    debug_msg[gvar.name_hint] = checker.total_weighted_compute_count
                    global_vars_to_inline.append(gvar)
                else:
                    all_small_subgraph = False

        if all_small_subgraph:
            return mod

        # do inline graph pass
        for k, v in debug_msg.items():
            DEBUG(f"{k} will be inlined to cpu with compute count:{v}")
        update_mod = relay.transform.InlineCompilerFunctionsBoundTo(global_vars_to_inline)(mod)

        return update_mod
