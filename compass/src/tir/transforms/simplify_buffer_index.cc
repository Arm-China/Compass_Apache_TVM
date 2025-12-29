// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/src/tir/transforms/simplify_buffer_index.cc
 */
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

class BufferIndexSimplifier : public StmtExprMutator {
  // Override methods that inherited from StmtMutator.
 public:
  Stmt VisitStmt_(const BufferStoreNode* op) final;
  Stmt VisitStmt_(const BufferRealizeNode* op) final;
  Stmt VisitStmt_(const AttrStmtNode* op) final;

  // Override methods that inherited from ExprMutator.
 public:
  PrimExpr VisitExpr_(const BufferLoadNode* op) final;

  // Methods of current class.
  // Internal supporting.
 private:
  Map<Buffer, Array<Var>> need_remove_vars_;
};

Array<PrimExpr> RemoveVars(Array<PrimExpr> exprs, Array<Var> vars) {
  Map<Var, PrimExpr> replace_map;
  for (const Var& var : vars) {
    replace_map.Set(var, 0);
  }

  Array<PrimExpr> ret;
  for (const PrimExpr& expr : exprs) {
    ret.push_back(Substitute(expr, replace_map));
  }
  return ret;
}

Stmt BufferIndexSimplifier::VisitStmt_(const BufferStoreNode* op) {
  Stmt ret = StmtExprMutator::VisitStmt_(op);
  auto iter = need_remove_vars_.find(op->buffer);
  if (iter == need_remove_vars_.end()) return ret;

  op = ret.as<BufferStoreNode>();
  return BufferStore(op->buffer, op->value, RemoveVars(op->indices, (*iter).second));
}

Stmt BufferIndexSimplifier::VisitStmt_(const BufferRealizeNode* op) {
  Array<Var> remove_vars;
  Array<Range> new_bounds;
  for (const Range& r : op->bounds) {
    PrimExpr new_min = r->min;
    if ((is_zero(r->min) == false) && is_positive_const(r->extent)) {
      // E.g. For A: [14, 4], AL is result of applying "cache_read" to A.
      // After applying "compute_at", obviously the "i" part of
      // "AL[(j + (i*7)), k]" need to be removed.
      //
      // for (i: 0, 2) {
      //   realize(AL, [(i*7):((i*7) + 7), 0:4], True {
      //     for (j: 0, 7) {
      //       for (k: 0, 4) {
      //         AL[(j + (i*7)), k] = A[(j + (i*7)), k]
      //       }
      //     }
      Array<Var> vars = UndefinedVars(r->min);
      ICHECK(vars.empty() == false);
      remove_vars = Concat(remove_vars, vars);
      new_min = 0;
    }
    new_bounds.push_back(Range::FromMinExtent(new_min, r->extent));
  }

  if (remove_vars.empty() == false) {
    need_remove_vars_.Set(op->buffer, remove_vars);
  }

  Stmt ret = StmtExprMutator::VisitStmt_(op);
  op = ret.as<BufferRealizeNode>();

  return BufferRealize(op->buffer, new_bounds, op->condition, op->body);
}

Stmt BufferIndexSimplifier::VisitStmt_(const AttrStmtNode* op) {
  Stmt ret = StmtExprMutator::VisitStmt_(op);
  // Handle buffer_bind_scope attr that tensorizing introduced.
  if (op->attr_key == attr::buffer_bind_scope) {
    Array<ObjectRef> arr = Downcast<Array<ObjectRef>>(op->node);
    CHECK_EQ(arr.size(), 2U);
    const BufferNode* buffer = arr[0].as<BufferNode>();
    const BufferNode* target = arr[1].as<BufferNode>();
    const CallNode* tuple = op->value.as<CallNode>();
    CHECK(buffer && target);
    CHECK(tuple && tuple->op.same_as(builtin::tvm_tuple()));

    auto iter = need_remove_vars_.find(GetRef<Buffer>(target));
    if (iter == need_remove_vars_.end()) return ret;

    op = ret.as<AttrStmtNode>();
    Array<PrimExpr> nargs = RemoveVars(tuple->args, (*iter).second);
    auto ntuple = Call(DataType::Handle(), builtin::tvm_tuple(), nargs);
    return AttrStmt(arr, op->attr_key, ntuple, op->body);
  }
  return ret;
}

PrimExpr BufferIndexSimplifier::VisitExpr_(const BufferLoadNode* op) {
  PrimExpr ret = StmtExprMutator::VisitExpr_(op);
  auto iter = need_remove_vars_.find(op->buffer);
  if (iter == need_remove_vars_.end()) return ret;

  op = ret.as<BufferLoadNode>();
  return BufferLoad(op->buffer, RemoveVars(op->indices, (*iter).second));
}

namespace transform {

// Simplify indices of "BufferLoad" and "BufferStore" by removing useless
// iteration variables before applying pass "tir.StorageFlatten".
Pass SimplifyBufferIndex() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = BufferIndexSimplifier()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.SimplifyBufferIndex", {});
}

TVM_FFI_REGISTER_GLOBAL("compass.tir.transform.SimplifyBufferIndex")
    .set_body_typed(SimplifyBufferIndex);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
