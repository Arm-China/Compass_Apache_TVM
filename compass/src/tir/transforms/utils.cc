// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/src/tir/transforms/utils.cc
 */
#include <compass/tvm/utils.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>

namespace tvm {
namespace tir {

bool is_all_true_pred(Optional<PrimExpr> predicate) {
  // This function may be called from any time, so there is many different scenarios.
  // The scenario that before running pass "LowerStandard".
  if (!predicate.defined()) return true;
  PrimExpr pred = predicate.value();

  if (pred.as<Bool>()) return Downcast<Bool>(pred) == true;

  if (!pred->IsInstance<CallNode>()) return false;

  auto call = Downcast<Call>(pred);
  if (call->op.same_as(tir::builtin::const_pred())) {
    // The scenario that before running pass "LowerPred".
    for (auto& arg : call->args) {
      if (Downcast<Bool>(arg) == false) return false;
    }
    return true;
  } else if (call->op.same_as(tir::builtin::call_extern())) {
    // The scenario that after running pass "LowerPred".
    std::string func_name = Downcast<StringImm>(call->args[0])->value;
    if (!StrStartsWith(func_name, "__vmov_") || !call->args[1]->IsInstance<IntImmNode>()) {
      return false;
    }
    int64_t pred_value = Downcast<IntImm>(call->args[1])->value;
    int lanes = call->dtype.lanes();
    for (int i = 0; i < lanes; ++i) {
      if ((pred_value & (1 << i * (32 / lanes))) == 0) return false;
    }
    return true;
  } else if (call->op.same_as(Op::Get("tir.precodegen"))) {
    // The scenario that after running pass "Precodegen".
    if (StrStartsWith(Downcast<StringImm>(call->args[0])->value, "ALL_TRUE")) return true;
    return false;
  }
  return false;
}

TVM_FFI_REGISTER_GLOBAL("compass.tir.transform.is_all_true_pred").set_body_typed(is_all_true_pred);

}  // namespace tir
}  // namespace tvm
