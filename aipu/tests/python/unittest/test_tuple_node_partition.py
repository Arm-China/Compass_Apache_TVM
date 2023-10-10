# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.
import tvm
from tvm import relay
from tvm.relay.backend.contrib.aipu_compass import transform as compass_transform


@tvm.ir.register_op_attr("concatenate", "target.test_tuple_partition", level=11)
def concatenate(expr):  # pylint: disable=unused-variable
    return True


@tvm.ir.register_op_attr("abs", "target.test_tuple_partition", level=11)
def check_abs(expr):  # pylint: disable=unused-variable
    return False


def test_var_input():
    inp0 = relay.var("inp0", relay.TensorType([1, 4], "float32"))
    inp1 = relay.var("inp1", relay.TensorType([1, 4], "float32"))
    concat = relay.concatenate([inp0, inp1], axis=0)
    func = relay.Function([inp0, inp1], concat)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    passes = [
        relay.transform.AnnotateTarget("test_tuple_partition"),
        compass_transform.ReAnnotateTuple(),
        relay.transform.MergeCompilerRegions(),
        relay.transform.PartitionGraph(),
    ]
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.transform.Sequential(passes)(mod)

    expect_main = """\
#[version = "0.0.5"]
fn (%inp0: Tensor[(1, 4), float32] /* ty=Tensor[(1, 4), float32] */, %inp1: Tensor[(1, 4), float32] /* ty=Tensor[(1, 4), float32] */) -> Tensor[(2, 4), float32] {
  @tvmgen_default_test_tuple_partition_main_0(%inp0, %inp1) /* ty=Tensor[(2, 4), float32] */
} /* ty=fn (Tensor[(1, 4), float32], Tensor[(1, 4), float32]) -> Tensor[(2, 4), float32] */
"""
    expect_subgraph0 = """\
#[version = "0.0.5"]
fn (%test_tuple_partition_0_i0: Tensor[(1, 4), float32] /* ty=Tensor[(1, 4), float32] */, %test_tuple_partition_0_i1: Tensor[(1, 4), float32] /* ty=Tensor[(1, 4), float32] */, Compiler="test_tuple_partition", Primitive=1, Inline=1, global_symbol="tvmgen_default_test_tuple_partition_main_0") -> Tensor[(2, 4), float32] {
  %0 = (%test_tuple_partition_0_i0, %test_tuple_partition_0_i1) /* ty=(Tensor[(1, 4), float32], Tensor[(1, 4), float32]) */;
  concatenate(%0) /* ty=Tensor[(2, 4), float32] */
} /* ty=fn (Tensor[(1, 4), float32], Tensor[(1, 4), float32]) -> Tensor[(2, 4), float32] */
"""

    assert mod["main"].astext() == expect_main.strip()
    assert mod.get_global_var("tvmgen_default_test_tuple_partition_main_0")
    assert mod["tvmgen_default_test_tuple_partition_main_0"].astext() == expect_subgraph0.strip()


def test_expr_input():
    inp0 = relay.var("inp0", relay.TensorType([1, 4], "float32"))
    inp1 = relay.var("inp1", relay.TensorType([1, 4], "float32"))
    abs0 = relay.abs(inp0)
    abs1 = relay.abs(inp1)
    concat = relay.concatenate([abs0, abs1], axis=0)
    func = relay.Function([inp0, inp1], concat)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    passes = [
        relay.transform.AnnotateTarget("test_tuple_partition"),
        compass_transform.ReAnnotateTuple(),
        relay.transform.MergeCompilerRegions(),
        relay.transform.PartitionGraph(),
    ]
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.transform.Sequential(passes)(mod)

    expect_main = """\
#[version = "0.0.5"]
fn (%inp0: Tensor[(1, 4), float32] /* ty=Tensor[(1, 4), float32] */, %inp1: Tensor[(1, 4), float32] /* ty=Tensor[(1, 4), float32] */) -> Tensor[(2, 4), float32] {
  %0 = abs(%inp0) /* ty=Tensor[(1, 4), float32] */;
  %1 = abs(%inp1) /* ty=Tensor[(1, 4), float32] */;
  @tvmgen_default_test_tuple_partition_main_0(%0, %1) /* ty=Tensor[(2, 4), float32] */
} /* ty=fn (Tensor[(1, 4), float32], Tensor[(1, 4), float32]) -> Tensor[(2, 4), float32] */
"""
    expect_subgraph0 = """\
#[version = "0.0.5"]
fn (%test_tuple_partition_0_i0: Tensor[(1, 4), float32] /* ty=Tensor[(1, 4), float32] */, %test_tuple_partition_0_i1: Tensor[(1, 4), float32] /* ty=Tensor[(1, 4), float32] */, Compiler="test_tuple_partition", Primitive=1, Inline=1, global_symbol="tvmgen_default_test_tuple_partition_main_0") -> Tensor[(2, 4), float32] {
  %0 = (%test_tuple_partition_0_i0, %test_tuple_partition_0_i1) /* ty=(Tensor[(1, 4), float32], Tensor[(1, 4), float32]) */;
  concatenate(%0) /* ty=Tensor[(2, 4), float32] */
} /* ty=fn (Tensor[(1, 4), float32], Tensor[(1, 4), float32]) -> Tensor[(2, 4), float32] */
"""

    assert mod["main"].astext() == expect_main.strip()
    assert mod.get_global_var("tvmgen_default_test_tuple_partition_main_0")
    assert mod["tvmgen_default_test_tuple_partition_main_0"].astext() == expect_subgraph0.strip()


if __name__ == "__main__":
    test_var_input()
    test_expr_input()
