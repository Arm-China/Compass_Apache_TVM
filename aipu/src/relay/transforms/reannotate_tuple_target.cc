// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file reannotate_tuple_target.cc
 */

#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "../../../../src/relay/transforms/pass_utils.h"

namespace tvm {
namespace relay {

static const PackedFunc* make_begin_op =
    runtime::Registry::Get("relay.op.annotation._make.compiler_begin");
static const PackedFunc* make_end_op =
    runtime::Registry::Get("relay.op.annotation._make.compiler_end");
static const char default_target[] = "default";

// Set the tuple annotation target to be the same as its user.
class ReAnnotateTupleRewriter : public ExprRewriter {
 public:
  ReAnnotateTupleRewriter() = default;

  Expr InsertAnnotation(const Expr& expr, const std::string& target, const PackedFunc* ann_op) {
    Expr new_op = (*ann_op)(expr, target);
    new_op->checked_type_ = expr->checked_type_;
    return new_op;
  }

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    if (call->op != CompilerBeginOp() ||
        call->attrs.as<CompilerAttrs>()->compiler == default_target) {
      return post;
    }
    String target = call->attrs.as<CompilerAttrs>()->compiler;
    const CallNode* end = Downcast<Call>(post)->args[0].as<CallNode>();
    if (!end || end->op != CompilerEndOp() ||
        end->attrs.as<CompilerAttrs>()->compiler != default_target) {
      return post;
    }
    Array<Expr> inputs;
    const TupleNode* tuple_node = end->args[0].as<TupleNode>();
    if (tuple_node) {
      for (auto f : tuple_node->fields) {
        const CallNode* begin = Downcast<Call>(f).as<CallNode>();
        if (!begin || begin->op != CompilerBeginOp() ||
            begin->attrs.as<CompilerAttrs>()->compiler != default_target) {
          return post;
        }
        inputs.push_back(begin->args[0]);
      }
    }
    if (!inputs.empty()) {
      Array<Expr> new_begins;
      for (auto inp : inputs) {
        new_begins.push_back(InsertAnnotation(inp, target, make_begin_op));
      }
      auto tuple = GetRef<Tuple>(tuple_node);
      auto new_tuple = WithFields(tuple, new_begins);
      Expr new_end = InsertAnnotation(new_tuple, target, make_end_op);
      return InsertAnnotation(new_end, target, make_begin_op);
    }
    return post;
  }
};

Expr ReAnnotateTuple(const Expr& expr) {
  ReAnnotateTupleRewriter reannotate_tuple;
  return PostOrderRewrite(expr, &reannotate_tuple);
}

namespace transform {

Pass ReAnnotateTuple() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(ReAnnotateTuple(f));
      };
  return CreateFunctionPass(pass_func, 1, "ReAnnotateTuple", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.ReAnnotateTuple").set_body_typed(ReAnnotateTuple);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
