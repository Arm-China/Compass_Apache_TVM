// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/src/relay/op/gruv3.cc
 * \brief Operator definitions for the AIPU Compass GRUv3.
 */

#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

namespace tvm {
namespace relay {

/*! \brief Attributes used in GRUv3 operator */
struct GRUv3Attrs : public tvm::AttrsNode<GRUv3Attrs> {
  std::string out_sequence;
  std::string activations;
  TVM_DECLARE_ATTRS(GRUv3Attrs, "relay.attrs.aipu_compass.GRUv3Attrs") {
    TVM_ATTR_FIELD(out_sequence).describe("The GRUv3 outputs.");
    TVM_ATTR_FIELD(activations).describe("The GRUv3 activation functions.");
  }
};  // struct GRUv3Attrs

TVM_REGISTER_NODE_TYPE(GRUv3Attrs);

Expr MakeGRUv3(Expr data, Expr init_state, Expr weights, Expr biases, std::string out_sequence,
               std::string activations) {
  auto attrs = make_object<GRUv3Attrs>();
  attrs->out_sequence = std::move(out_sequence);
  attrs->activations = std::move(activations);
  static const Op& op = Op::Get("contrib.aipu_compass.gruv3");
  return Call(op, {data, init_state, weights, biases}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relay.op.contrib.aipu_compass._make.gruv3").set_body_typed(MakeGRUv3);

bool GRUv3Rel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
              const TypeReporter& reporter) {
  static auto op_rel = tvm::runtime::Registry::Get("relay.op.contrib.aipu_compass.gruv3_rel");
  ICHECK(op_rel) << "relay.op.contrib.aipu_compass.gruv3_rel isn't registered.";
  const auto* param = attrs.as<GRUv3Attrs>();
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  const auto* init_state = types[1].as<TensorTypeNode>();
  if (init_state == nullptr) return false;
  const auto* weights = types[2].as<TensorTypeNode>();
  if (weights == nullptr) return false;
  const auto* biases = types[3].as<TensorTypeNode>();
  if (biases == nullptr) return false;

  auto ret = (*op_rel)(types, param->out_sequence, param->activations);
  if (ret.type_code() == kTVMStr) {
    std::string reason = ret;
    reporter->GetDiagCtx().EmitFatal(
        Diagnostic::Error(reporter->GetSpan())
        << "Failed to call the relay.op.contrib.aipu_compass.gruv3_rel " << reason);
    return false;
  }

  Array<TensorType> ret_type = ret;
  for (uint32_t i = 0; i < ret_type.size(); i++) {
    reporter->Assign(types[4 + i], ret_type[i]);
  }

  return true;
}

RELAY_REGISTER_OP("contrib.aipu_compass.gruv3")
    .describe(R"code(AIPU Compass GRUv3 operator.)code" TVM_ADD_FILELINE)
    .set_num_inputs(4)
    .set_support_level(5)
    .add_argument("data", "Tensor", "The input sequence of GRUv3.")
    .add_argument("init_state", "Tensor", "The init state of GRUv3.")
    .add_argument("weights", "Tensor", "The weights of GRUv3.")
    .add_argument("biases", "Tensor", "The biases of GRUv3.")
    .set_attrs_type<GRUv3Attrs>()
    .add_type_rel("GRUv3", GRUv3Rel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace relay
}  // namespace tvm
