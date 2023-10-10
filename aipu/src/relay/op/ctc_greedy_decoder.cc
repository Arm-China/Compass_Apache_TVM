// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/src/relay/op/ctc_greedy_decoder.cc
 * \brief Operator definitions for CTCGreedyDecoder.
 */

#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/topi/detail/constant_utils.h>

namespace tvm {
namespace relay {

/*! \brief Attributes used in CTCGreedyDecoder operator */
struct CTCGreedyDecoderAttrs : public tvm::AttrsNode<CTCGreedyDecoderAttrs> {
  bool merge_repeated;

  TVM_DECLARE_ATTRS(CTCGreedyDecoderAttrs, "relay.attrs.aipu_compass.CTCGreedyDecoderAttrs") {
    TVM_ATTR_FIELD(merge_repeated)
        .set_default(true)
        .describe("Whether merge repeated classes in output.");
  }
};  // struct CTCGreedyDecoderAttrs

TVM_REGISTER_NODE_TYPE(CTCGreedyDecoderAttrs);

Expr MakeCTCGreedyDecoder(Expr data, Expr seq_len, bool merge_repeated) {
  auto attrs = make_object<CTCGreedyDecoderAttrs>();
  attrs->merge_repeated = merge_repeated;
  static const Op& op = Op::Get("contrib.aipu_compass.ctc_greedy_decoder");
  return Call(op, {data, seq_len}, Attrs(attrs), {});
}

bool CTCGreedyDecoderRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                         const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);

  const auto* data = types[0].as<TensorTypeNode>();
  const auto* seq_len = types[1].as<TensorTypeNode>();
  if (!data || !seq_len) return false;

  ICHECK(tvm::topi::detail::EqualCheck(data->shape[0], seq_len->shape[0]))
      << "batch_size should be same.";
  Array<IndexExpr> oshape;
  // Here follow the rule of parser & opt & lib.
  // For really shape is dynamic, will insert -1 at tail.
  oshape.push_back(data->shape[0]);
  oshape.push_back(4096);
  oshape.push_back(1);
  oshape.push_back(1);
  reporter->Assign(types[2], TensorType(oshape, DataType::Int(32)));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.contrib.aipu_compass._make.ctc_greedy_decoder")
    .set_body_typed(MakeCTCGreedyDecoder);

RELAY_REGISTER_OP("contrib.aipu_compass.ctc_greedy_decoder")
    .describe(R"code(TensorFlow CTCGreedyDecoder operator.)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("seq_len", "Tensor", "1-D int32 vector containing sequence lengths.")
    .set_attrs_type<CTCGreedyDecoderAttrs>()
    .set_support_level(5)
    .add_type_rel("CTCGreedyDecoder", CTCGreedyDecoderRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace relay
}  // namespace tvm
