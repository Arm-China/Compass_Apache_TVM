// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/src/relay/op/channel_shuffle.cc
 * \brief Operator definitions for ChannelShuffle.
 */

#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

#include "../../../../src/relay/op/op_common.h"

namespace tvm {
namespace relay {

/*! \brief Attributes used in ChannelShuffle operator */
struct ChannelShuffleAttrs : public tvm::AttrsNode<ChannelShuffleAttrs> {
  int group;
  int splits;

  TVM_DECLARE_ATTRS(ChannelShuffleAttrs, "relay.attrs.aipu_compass.ChannelShuffleAttrs") {
    TVM_ATTR_FIELD(group).describe("The group number of the input tensor.");
    TVM_ATTR_FIELD(splits).describe("The number of output tensors.");
  }
};  // struct ChannelShuffleAttrs

TVM_REGISTER_NODE_TYPE(ChannelShuffleAttrs);

Expr MakeChannelShuffle(Expr data, int group, int splits) {
  auto attrs = make_object<ChannelShuffleAttrs>();
  attrs->group = group;
  attrs->splits = splits;
  static const Op& op = Op::Get("contrib.aipu_compass.channel_shuffle");
  return Call(op, {data}, Attrs(attrs), {});
}

bool ChannelShuffleRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                       const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  /*output shape is the same as input shape.*/
  Array<IndexExpr> oshape = data->shape;
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

InferCorrectLayoutOutput ChannelShuffleInferCorrectLayout(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types) {
  Layout ret;

  if (new_in_layouts.defined()) {
    ICHECK_GE(new_in_layouts.size(), 1);
    ret = new_in_layouts[0];
  } else if (old_in_layouts.defined()) {
    ICHECK_GE(old_in_layouts.size(), 1);
    ret = old_in_layouts[0];
  }

  return InferCorrectLayoutOutput({ret}, {ret}, attrs);
}

TVM_REGISTER_GLOBAL("relay.op.contrib.aipu_compass._make.channel_shuffle")
    .set_body_typed(MakeChannelShuffle);

RELAY_REGISTER_OP("contrib.aipu_compass.channel_shuffle")
    .describe(R"code(ChannelShuffle operator.)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attrs_type<ChannelShuffleAttrs>()
    .set_support_level(5)
    .add_type_rel("ChannelShuffle", ChannelShuffleRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ChannelShuffleInferCorrectLayout)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace relay
}  // namespace tvm
