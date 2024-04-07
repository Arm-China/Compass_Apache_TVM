// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file primal_layout_transform_to_transpose.cc
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/data_layout.h>

#include "../../../../src/relay/op/make_op.h"

namespace tvm {
namespace relay {

class PrimalLayoutTransformToTransposeAdaptor : public ExprRewriter {
 public:
  PrimalLayoutTransformToTransposeAdaptor() : layout_transform_op_(Op::Get("layout_transform")) {}

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    if (call->op == layout_transform_op_) {
      auto attrs = call->attrs.as<LayoutTransformAttrs>();
      Layout src_layout(attrs->src_layout);
      Layout dst_layout(attrs->dst_layout);
      auto input = post.as<CallNode>()->args[0];

      ICHECK(src_layout.defined() && dst_layout.defined())
          << "cannot convert from/to undefined layout";

      if (src_layout.Equals(dst_layout)) {
        return input;
      }

      for (auto src_axis : src_layout->axes) {
        if (!LayoutAxis::Get(src_axis).IsPrimal()) {
          return post;
        }
      }

      for (auto dst_axis : dst_layout->axes) {
        if (!LayoutAxis::Get(dst_axis).IsPrimal()) {
          return post;
        }
      }

      auto layout_converter = tir::BijectiveLayout(src_layout, dst_layout);
      ICHECK(layout_converter.defined())
          << "cannot convert from " << attrs->src_layout << " to " << attrs->dst_layout;

      Array<Integer> transpose_axes;
      for (auto dst_axis : dst_layout->axes) {
        uint32_t index = src_layout.IndexOf(LayoutAxis::Get(dst_axis));
        transpose_axes.push_back(index);
      }

      return MakeTranspose(input, transpose_axes);
    }
    return post;
  }

 private:
  const Op& layout_transform_op_;
};

Expr PrimalLayoutTransformToTranspose(const Expr& expr) {
  auto rewriter = PrimalLayoutTransformToTransposeAdaptor();
  return PostOrderRewrite(expr, &rewriter);
}

namespace transform {

Pass PrimalLayoutTransformToTranspose() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(PrimalLayoutTransformToTranspose(f));
      };
  return CreateFunctionPass(pass_func, 1, "PrimalLayoutTransformToTranspose", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.PrimalLayoutTransformToTranspose")
    .set_body_typed(PrimalLayoutTransformToTranspose);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
