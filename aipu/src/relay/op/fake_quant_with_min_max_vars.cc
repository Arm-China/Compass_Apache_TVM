// This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
// See the copyright file distributed with this work for additional information
// regarding copyright ownership.
/*!
 * \file aipu/src/relay/op/fake_quant_with_min_max_vars.cc
 * \brief Operator definitions for FakeQuantWithMinMaxVars.
 */

#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

namespace tvm {
namespace relay {

/*! \brief Attributes used in FakeQuantWithMinMaxVars operator */
struct FakeQuantWithMinMaxVarsAttrs : public tvm::AttrsNode<FakeQuantWithMinMaxVarsAttrs> {
  bool narrow_range;
  int num_bits;
  double minimum;
  double maximum;

  TVM_DECLARE_ATTRS(FakeQuantWithMinMaxVarsAttrs,
                    "relay.attrs.aipu_compass.FakeQuantWithMinMaxVarsAttrs") {
    TVM_ATTR_FIELD(narrow_range)
        .set_default(false)
        .describe("Quant to [0,2^num_bits - 1] when false and [1,2^num_bits - 1] when true.");
    TVM_ATTR_FIELD(num_bits).set_default(8).describe("The bitwidth of the quantization.");
    TVM_ATTR_FIELD(minimum).describe("The clip minimum for input.");
    TVM_ATTR_FIELD(maximum).describe("The clip minimum for input.");
  }
};  // struct FakeQuantWithMinMaxVarsAttrs

TVM_REGISTER_NODE_TYPE(FakeQuantWithMinMaxVarsAttrs);

Expr MakeFakeQuantWithMinMaxVars(Expr data, double minimum, double maximum, bool narrow_range,
                                 int num_bits) {
  auto attrs = make_object<FakeQuantWithMinMaxVarsAttrs>();
  attrs->narrow_range = narrow_range;
  attrs->num_bits = num_bits;
  attrs->minimum = minimum;
  attrs->maximum = maximum;
  static const Op& op = Op::Get("contrib.aipu_compass.fake_quant_with_min_max_vars");
  return Call(op, {data}, Attrs(attrs), {});
}

bool FakeQuantWithMinMaxVarsRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                                const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  /*output shape is the same as input shape.*/
  Array<IndexExpr> oshape = data->shape;
  reporter->Assign(types[1], TensorType(oshape, DataType::Float(32)));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.contrib.aipu_compass._make.fake_quant_with_min_max_vars")
    .set_body_typed(MakeFakeQuantWithMinMaxVars);

RELAY_REGISTER_OP("contrib.aipu_compass.fake_quant_with_min_max_vars")
    .describe(R"code(TensorFlow FakeQuantWithMinMaxVars operator.)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attrs_type<FakeQuantWithMinMaxVarsAttrs>()
    .set_support_level(5)
    .add_type_rel("FakeQuantWithMinMaxVars", FakeQuantWithMinMaxVarsRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace relay
}  // namespace tvm
