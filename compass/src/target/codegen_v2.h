// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/src/target/codegen_v2.h
 */
#ifndef COMPASS_TARGET_CODEGEN_V2_H_
#define COMPASS_TARGET_CODEGEN_V2_H_

#include <compass/tvm/target/compass_info.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "../../../../src/target/source/codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenCompassV2 final : public CodeGenC {
  // Override methods that inherited from CodeGenC.
 private:
  void InitFuncState(const PrimFunc& f) final;
  void PrintFuncPrefix(std::ostream& os) final;
  void BindThreadIndex(const IterVar& iv) final;
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;
  void PrintType(DataType t, std::ostream& os) final;
  void PrintType(const Type& type, std::ostream& os) final;
  void AddFunction(const GlobalVar& gvar, const PrimFunc& f) final;
  std::string Finish() final;
  std::string GetBufferRef(DataType dtype, const BufferNode* buffer, PrimExpr index) final;
  void PrintCallExtern(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                       bool skip_first_arg, std::ostream& os);

  // expression
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) final;
  void VisitExpr_(const CallNode* op, std::ostream& os) final;
  void VisitExpr_(const IntImmNode* op, std::ostream& os) final;
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) final;
  void VisitExpr_(const StringImmNode* op, std::ostream& os) final;

  // overload to avoid ambiguous call errors
  void VisitExpr_(const AddNode* op, std::ostream& os) final;
  void VisitExpr_(const SubNode* op, std::ostream& os) final;
  void VisitExpr_(const MulNode* op, std::ostream& os) final;
  void VisitExpr_(const DivNode* op, std::ostream& os) final;

  void VisitExpr_(const MinNode* op, std::ostream& os) final;
  void VisitExpr_(const MaxNode* op, std::ostream& os) final;
  void VisitExpr_(const AndNode* op, std::ostream& os) final;
  void VisitExpr_(const OrNode* op, std::ostream& os) final;
  void VisitExpr_(const SelectNode* op, std::ostream& os) final;
  void VisitExpr_(const CastNode* op, std::ostream& os) final;
  void VisitExpr_(const FloorDivNode* op, std::ostream& os) final;
  void VisitExpr_(const BufferLoadNode* op, std::ostream& os) final;

  // statment
  void VisitStmt_(const AllocateNode* op) final;
  void VisitStmt_(const BufferStoreNode* op) final;
  void VisitStmt_(const ForNode* op) final;
  void VisitStmt_(const LetStmtNode* op) final;
  void VisitStmt_(const AllocateConstNode* op) override;
  void VisitStmt_(const WhileNode* op) override;

  // Methods of current class.
 public:
  CodeGenCompassV2(CompassInfo cps_info);
  std::string PrintType(DataType t);

  // Internal supporting.
 private:
  // Information of the Zhouyi NPU configuration that the program is built to.
  CompassInfo cps_info_;
  bool is_entry_ = false;
  std::string attrs_ = "";
  std::unordered_set<std::string> macro_instances_;
  std::ostringstream dma_macro_decl_stream_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // COMPASS_TARGET_CODEGEN_V2_H_
