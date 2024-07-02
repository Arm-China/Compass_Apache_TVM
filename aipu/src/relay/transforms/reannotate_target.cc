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

Expr InsertAnnotation(const Expr& expr, const std::string& target, const PackedFunc* ann_op) {
  Expr new_op = (*ann_op)(expr, target);
  new_op->checked_type_ = expr->checked_type_;
  return new_op;
}

inline Expr InsertDefaultEnd(const Expr& expr) {
  return InsertAnnotation(expr, default_target, make_end_op);
}

inline Expr InsertDefaultBegin(const Expr& expr) {
  return InsertAnnotation(expr, default_target, make_begin_op);
}

// Set the tuple annotation target to be the same as its user.
class ReAnnotateTupleRewriter : public ExprRewriter {
 public:
  ReAnnotateTupleRewriter() = default;

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

// Inline function and update params to args.
class InlineFunction : public ExprRewriter {
 public:
  explicit InlineFunction(std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> param2arg)
      : param2arg_(std::move(param2arg)) {}

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    Array<Expr> new_args;
    for (Expr arg : post.as<CallNode>()->args) {
      if (arg.as<VarNode>() && param2arg_.find(arg) != param2arg_.end()) {
        new_args.push_back(param2arg_[arg]);
      } else {
        if (!arg.as<ConstantNode>()) {
          arg = InsertDefaultEnd(arg);
        }
        new_args.push_back(InsertDefaultBegin(arg));
      }
    }
    return Call(call->op, new_args, call->attrs, call->type_args, call->span);
  }

 protected:
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> param2arg_;
};

// Set exclude nodes annotation to default.
class AnnotateNodesToDefault : public ExprRewriter {
 public:
  explicit AnnotateNodesToDefault(std::unordered_set<Expr, ObjectPtrHash> exclude_nodes)
      : exclude_nodes_(std::move(exclude_nodes)) {}

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    if (call->op != CompilerEndOp()) {
      return post;
    }
    const Expr& inp0 = call->args[0];
    if (exclude_nodes_.find(inp0) == exclude_nodes_.end()) {
      return post;
    }
    if (inp2new_inp.find(inp0) != inp2new_inp.end()) {
      return InsertDefaultEnd(inp2new_inp[inp0]);
    }

    Expr new_inp;
    if (const CallNode* arg_call = inp0.as<CallNode>()) {
      Array<Expr> new_begins;
      for (const Expr& arg : arg_call->args) {
        const CallNode* bg = arg.as<CallNode>();
        ICHECK(bg && bg->op == CompilerBeginOp());
        new_begins.push_back(InsertDefaultBegin(bg->args[0]));
      }
      const FunctionNode* func_node = arg_call->op.as<FunctionNode>();
      if (func_node) {
        ICHECK_EQ(new_begins.size(), func_node->params.size());
        std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> param2arg;
        for (size_t i = 0; i < new_begins.size(); i++) {
          param2arg[func_node->params[i]] = new_begins[i];
        }
        InlineFunction inliner(param2arg);
        Expr new_func = PostOrderRewrite(arg_call->op, &inliner);
        new_inp = new_func.as<FunctionNode>()->body;
      } else {
        new_inp =
            Call(arg_call->op, new_begins, arg_call->attrs, arg_call->type_args, arg_call->span);
      }
    } else if (const TupleNode* arg_tup = inp0.as<TupleNode>()) {
      Array<Expr> new_begins;
      for (const Expr& f : arg_tup->fields) {
        const CallNode* bg = f.as<CallNode>();
        ICHECK(bg && bg->op == CompilerBeginOp());
        new_begins.push_back(InsertDefaultBegin(bg->args[0]));
      }
      new_inp = Tuple(new_begins, arg_tup->span);
    } else if (const TupleGetItemNode* arg_tgn = inp0.as<TupleGetItemNode>()) {
      const CallNode* bg = arg_tgn->tuple.as<CallNode>();
      ICHECK(bg && bg->op == CompilerBeginOp());
      Expr new_begin = InsertDefaultBegin(bg->args[0]);
      new_inp = TupleGetItem(new_begin, arg_tgn->index, arg_tgn->span);
    } else {
      LOG(FATAL) << "Need to support.";
    }

    inp2new_inp[inp0] = new_inp;
    return InsertDefaultEnd(new_inp);
  }

 protected:
  /*! \brief The nodes to update compiler attr to default. */
  std::unordered_set<Expr, ObjectPtrHash> exclude_nodes_;
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> inp2new_inp;
};

Expr ReAnnotateTuple(const Expr& expr) {
  ReAnnotateTupleRewriter reannotate_tuple;
  return PostOrderRewrite(expr, &reannotate_tuple);
}

Expr SetNodeCompilerToDefault(const Expr& expr, const Array<IntImm>& indices) {
  if (indices.size() == 0) {
    return expr;
  }
  auto func = tvm::runtime::Registry::Get("relay.analysis.PrinterIndexToExpr");
  Array<Expr> nodes = (*func)(Downcast<Function>(expr));
  std::unordered_set<Expr, ObjectPtrHash> exclude_nodes;
  for (const IntImm& index : indices) {
    exclude_nodes.insert(nodes[index.as<IntImmNode>()->value]);
  }

  AnnotateNodesToDefault rewriter(exclude_nodes);
  return PostOrderRewrite(expr, &rewriter);
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

Pass SetNodeCompilerToDefault(Array<IntImm> indices) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(SetNodeCompilerToDefault(f, indices));
      };
  return CreateFunctionPass(pass_func, 1, "SetNodeCompilerToDefault", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.SetNodeCompilerToDefault")
    .set_body_typed(SetNodeCompilerToDefault);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
