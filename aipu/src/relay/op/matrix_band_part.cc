// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/src/relay/op/matrix_band_part.cc
 * \brief Operator definitions for MatrixBandPart.
 */

#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

#include "../../../../src/relay/op/type_relations.h"

namespace tvm {
namespace relay {

/*! \brief Attributes used in MatrixBandPart operator */
struct MatrixBandPartAttrs : public tvm::AttrsNode<MatrixBandPartAttrs> {
  int num_lower;
  int num_upper;
  TVM_DECLARE_ATTRS(MatrixBandPartAttrs, "relay.attrs.aipu_compass.MatrixBandPartAttrs") {
    TVM_ATTR_FIELD(num_lower).describe(
        "0-D tensor. Number of subdiagonals to keep. If negative, keep entire lower triangle.");
    TVM_ATTR_FIELD(num_upper).describe(
        "0-D tensor. Number of superdiagonals to keep. If negative, keep entire upper triangle.");
  }
};  // struct MatrixBandPartAttrs

TVM_REGISTER_NODE_TYPE(MatrixBandPartAttrs);

Expr MakeMatrixBandPart(Expr data, int num_lower, int num_upper) {
  auto attrs = make_object<MatrixBandPartAttrs>();
  attrs->num_lower = num_lower;
  attrs->num_upper = num_upper;
  static const Op& op = Op::Get("contrib.aipu_compass.matrix_band_part");
  return Call(op, {data}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relay.op.contrib.aipu_compass._make.matrix_band_part")
    .set_body_typed(MakeMatrixBandPart);

RELAY_REGISTER_OP("contrib.aipu_compass.matrix_band_part")
    .describe(R"code(TensorFlow MatrixBandPart operator.)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_support_level(5)
    .add_argument("input", "Tensor", "The input matrix.")
    .set_attrs_type<MatrixBandPartAttrs>()
    .add_type_rel("MatrixBandPart", IdentityRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace relay
}  // namespace tvm
