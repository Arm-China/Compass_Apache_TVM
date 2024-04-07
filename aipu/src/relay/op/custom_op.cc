// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/src/relay/op/custom_op.cc
 * \brief Operator definitions for CustomOp.
 */

#include <tvm/ir/attrs.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

namespace tvm {
namespace relay {

/*! \brief Attributes used in CustomOp operator */
struct CustomOpAttrs : public tvm::AttrsNode<CustomOpAttrs> {
  Type out_type;
  TVM_DECLARE_ATTRS(CustomOpAttrs, "relay.attrs.aipu_compass.CustomOpAttrs") {
    TVM_ATTR_FIELD(out_type).describe("The output type of the operator.");
  }
};  // struct CustomOpAttrs

TVM_REGISTER_NODE_TYPE(CustomOpAttrs);

Expr MakeCustomOp(Expr data, Type out_type) {
  auto attrs = make_object<CustomOpAttrs>();
  attrs->out_type = out_type;
  static const Op& op = Op::Get("contrib.aipu_compass.custom_op");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.contrib.aipu_compass._make.custom_op").set_body_typed(MakeCustomOp);

bool CustomOpRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  // If the input or output count > 1, the corresponding type must be TupleType.
  ICHECK(types.size() == 2 && num_inputs == 1);
  Type out_type = attrs.as<CustomOpAttrs>()->out_type;
  if (out_type.defined()) {
    reporter->Assign(types.back(), out_type);
    return true;
  }
  return false;
}

RELAY_REGISTER_OP("contrib.aipu_compass.custom_op")
    .describe(
        "The customized operator with any input or output count, and fixed type "
        "relation." TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Union[Tensor, Tuple[Tensor]]",
                  "The single input or the tuple that contains all inputs.")
    .set_attrs_type<CustomOpAttrs>()
    .add_type_rel("CustomOp", CustomOpRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace relay
}  // namespace tvm
