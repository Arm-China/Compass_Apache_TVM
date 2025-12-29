// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/src/relax/op/channel_shuffle.cc
 * \brief Operator definitions for ChannelShuffle.
 */
#include <tvm/relax/expr.h>

#include <utility>
#include <vector>

#include "../../../../src/relax/op/op_common.h"

namespace tvm {
namespace relax {

/*! \brief Attributes used in ChannelShuffle operator */
struct ChannelShuffleAttrs : public tvm::AttrsNode<ChannelShuffleAttrs> {
  int group;
  int axis;
  int splits;

  TVM_DECLARE_ATTRS(ChannelShuffleAttrs, "relax.attrs.compass.ChannelShuffleAttrs") {
    TVM_ATTR_FIELD(group).describe("The group number of the input tensor.");
    TVM_ATTR_FIELD(axis).describe("The dimension along which to shuffle.");
    TVM_ATTR_FIELD(splits).describe("The number of output tensors.");
  }
};  // struct ChannelShuffleAttrs

/* relax.channel_shuffle */
TVM_REGISTER_NODE_TYPE(ChannelShuffleAttrs);

Expr channel_shuffle(Expr data, int group, int axis, int splits) {
  auto attrs = make_object<ChannelShuffleAttrs>();
  attrs->group = group;
  attrs->axis = axis;
  attrs->splits = splits;
  static const Op& op = Op::Get("relax.channel_shuffle");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_FFI_REGISTER_GLOBAL("relax.op.compass.channel_shuffle").set_body_typed(channel_shuffle);

InferLayoutOutput InferLayoutChannelShuffle(const Call& call,
                                            const Map<String, Array<String>>& desired_layouts,
                                            const VarLayoutMap& var_layout_map) {
  const auto& it = desired_layouts.find("relax.channel_shuffle");
  const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(call);
  ICHECK(tensor_sinfo != nullptr) << "Invalid Call";
  ICHECK_EQ(tensor_sinfo->ndim, 4) << "Unsupported initial layout";
  const auto* attrs = call->attrs.as<ChannelShuffleAttrs>();
  ICHECK(attrs) << "Invalid Call";
  ICHECK(attrs->axis == 3 || attrs->axis == 1) << "Invalid axis: " << attrs->axis;
  std::string input_layout{(attrs->axis == 1) ? "NCHW" : "NHWC"};

  LayoutDecision layout;
  ObjectPtr<ChannelShuffleAttrs> new_attrs = make_object<ChannelShuffleAttrs>(*attrs);

  if (it != desired_layouts.end()) {
    // We have a desired layout.
    Layout desired_data_layout = (*it).second[0];
    ICHECK_EQ(desired_data_layout.ndim(), desired_data_layout.ndim_primal()) << "Axis swap only";
    layout = TransposeLike(InitialLayout(4), input_layout, desired_data_layout);
    new_attrs->axis = FindAxis(layout->layout, attrs->axis);
  } else {
    // We dont have a desired layout, propagate from the input instead.
    layout = GetLayoutDecision(var_layout_map, call->args[0]);
    if (layout->layout.ndim() != layout->layout.ndim_primal()) {
      tir::Layout in_layout(input_layout, DataType::Int(64));
      auto desired_layout = TransposeSubLayoutLike(input_layout, InitialLayout(4), layout->layout);
      auto data_si = GetStructInfo(call->args[0]);
      TensorStructInfo data_sinfo = data_si.as<TensorStructInfo>().value();
      Optional<ShapeExpr> data_shape = GetRef<ShapeExpr>(data_sinfo->shape.as<ShapeExprNode>());
      if (CanProveLayoutTransform(in_layout, desired_layout, data_shape.value()->values)) {
        new_attrs->axis = FindAxis(desired_layout, attrs->axis);
        return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
      } else {
        layout = InitialLayout(4);
      }
    }

    new_attrs->axis = FindAxis(layout->layout, attrs->axis);
  }
  return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
}

TVM_REGISTER_OP("relax.channel_shuffle")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attrs_type<ChannelShuffleAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                InferStructInfoUnaryArith</*require_float_dtype=*/false>)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutChannelShuffle)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
