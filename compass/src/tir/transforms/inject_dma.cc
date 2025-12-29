// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/src/tir/transforms/inject_dma.cc
 */
#include <compass/tvm/target/compass_info.h>
#include <compass/tvm/utils.h>
#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../../src/tir/transforms/ir_utils.h"

namespace tvm {
namespace tir {

using arith::Analyzer;
using arith::ConstIntBound;
using arith::DetectLinearEquation;

class DmaInjector : public StmtExprMutator {
  // Override methods that inherited from StmtMutator.
 public:
  Stmt VisitStmt_(const AttrStmtNode* op) final;

  // Methods of current class.
  // Internal supporting.
 private:
  enum DmaDirection { kDdr2Sram = 0, kSram2Ddr = 1 };
  PrimExpr AbsoluteAddressOf(Buffer buf, PrimExpr index, PrimExpr tec_idx) const;
};

static inline String GetStorageScope(Var var) {
  String scope = GetPtrStorageScope(var);
  if (scope == "") scope = "global";
  return scope;
}

PrimExpr DmaInjector::AbsoluteAddressOf(Buffer buf, PrimExpr offset, PrimExpr tec_idx) const {
  // The "cast" node is used to avoid Compass C compiler report conversion warning.
  auto addr = Cast(DataType::UInt(32),
                   Call(DataType::Handle(), builtin::address_of(), {BufferLoad(buf, {offset})}));
  String scope = GetStorageScope(buf->data);
  if (scope == "global") return addr;

  // TODO(compass-team): Can't ensure whether code generation of "address_of" can
  // support the non-zero offset when the variable type is vector, so just error
  // out now.
  ICHECK(is_zero(offset));

  // For SRAM scenario, the address above actually is just offset, the base
  // address of corresponding SRAM need to be added.
  String func_name = "__get_" + StrReplace(scope, ".", "") + "_addr";
  Array<PrimExpr> args{StringImm(func_name), addr};
  if (StrStartsWith(scope, "lsram") == true) {
    args.push_back(tec_idx);
  }
  return Call(DataType::UInt(32), builtin::call_extern(), args);
}

// For example, if polynomial is "ax + by + cz + d" and vars is "[x, y]", then
// the result is "cz + d".
inline PrimExpr GetConstantCoefficient(PrimExpr polynomial, Array<Var> vars) {
  Array<PrimExpr> coefficients = DetectLinearEquation(polynomial, vars);
  ICHECK(coefficients.size() == (vars.size() + 1));
  return coefficients[vars.size()];
}

// For example, if coeff is "[a, b]" and vars is "[x, y]", then
// the result is "ax + by".
inline PrimExpr RestructIndice(Array<PrimExpr> coeff, Array<Var> vars) {
  ICHECK_EQ(coeff.size(), vars.size() + 1);
  PrimExpr indice = make_zero(DataType::Int(32));
  for (size_t i = 0; i < vars.size(); ++i) {
    indice = indice + coeff[i] * vars[i];
  }
  return indice;
}

Stmt DmaInjector::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == "pragma_compass_dma_copy") {
    Array<Var> itervars;
    PrimExpr data_count = 1;
    Stmt cur_stmt = op->body;
    while (const auto* node = cur_stmt.as<ForNode>()) {
      ICHECK(is_zero(node->min));
      itervars.push_back(node->loop_var);
      data_count *= node->extent;
      cur_stmt = node->body;
    }
    const auto* store = cur_stmt.as<BufferStoreNode>();
    ICHECK((store != nullptr) && is_all_true_pred(store->predicate));
    const auto* load = store->value.as<BufferLoadNode>();
    ICHECK((load != nullptr) && is_all_true_pred(load->predicate));

    // Get the base offset of the "load" and "store" node by eliminating all of
    // iteration variables of the "for" nodes.
    PrimExpr store_offset = GetConstantCoefficient(store->indices[0], itervars);
    PrimExpr load_offset = GetConstantCoefficient(load->indices[0], itervars);

    PrimExpr dest_addr = AbsoluteAddressOf(store->buffer, store_offset, op->value);
    PrimExpr src_addr = AbsoluteAddressOf(load->buffer, load_offset, op->value);

    DmaDirection direction = kDdr2Sram;
    if (GetStorageScope(store->buffer->data) == "global") {
      direction = kSram2Ddr;
    }

    return Evaluate(Call(DataType::Void(), builtin::call_extern(),
                         {StringImm("tvm_cfgdma"), dest_addr, src_addr,
                          data_count * load->dtype.bytes(), direction}));
  }

  return StmtExprMutator::VisitStmt_(op);
}

class DmaInjectorV2 : public StmtExprMutator {
  // Override methods that inherited from StmtMutator.
 public:
  Stmt VisitStmt_(const AttrStmtNode* op) final;

  // Methods of current class.
  // Internal supporting.
 private:
  // Internal analzyer
  Analyzer analyzer_;
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> loop_var_map_;
};

Stmt DmaInjectorV2::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == "pragma_compass_dma_copy") {
    Array<Var> loop_vars;
    Stmt cur_stmt = op->body;
    while (const auto* node = cur_stmt.as<ForNode>()) {
      auto loop_var = node->loop_var;
      loop_vars.push_back(loop_var);
      loop_var_map_[loop_var] = node->extent - 1;
      cur_stmt = node->body;
    }
    const auto* store = cur_stmt.as<BufferStoreNode>();
    ICHECK((store != nullptr) && is_all_true_pred(store->predicate));
    const auto* load = store->value.as<BufferLoadNode>();
    ICHECK((load != nullptr) && is_all_true_pred(load->predicate));
    ICHECK((store->indices.size() == 1) && (load->indices.size() == 1));

    Array<PrimExpr> store_strides = DetectLinearEquation(store->indices[0], loop_vars);
    ICHECK(store_strides.size() == (loop_vars.size() + 1));
    Array<PrimExpr> load_strides = DetectLinearEquation(load->indices[0], loop_vars);
    ICHECK(load_strides.size() == (loop_vars.size() + 1));

    // Get the base offset of the "load" and "store" node by eliminating all of
    // iteration variables of the "for" nodes.
    PrimExpr store_offset = store_strides[loop_vars.size()];
    PrimExpr load_offset = load_strides[loop_vars.size()];

    PrimExpr dest_addr =
        Call(DataType::Handle(), builtin::pointer(),
             {tir::TypeAnnotation(store->value.dtype()), StringImm(store->buffer.scope()),
              store->buffer.get()->data, store_offset});
    PrimExpr src_addr = Call(DataType::Handle(), builtin::pointer(),
                             {tir::TypeAnnotation(load->dtype), StringImm(load->buffer.scope()),
                              load->buffer.get()->data, load_offset});

    // Always consider sram is continous.
    // load and store indice is equal, generate DMA1D.
    if (analyzer_.CanProveEqual(RestructIndice(store_strides, loop_vars),
                                RestructIndice(load_strides, loop_vars))) {
      PrimExpr data_size = make_zero(DataType::Int(32));
      for (size_t i = 0; i < loop_vars.size(); ++i) {
        data_size += loop_var_map_[loop_vars[i]] * load_strides[i];
      }
      // Add 1 for count 0.
      data_size += make_const(DataType::Int(32), 1);
      PrimExpr data_count = analyzer_.Simplify(data_size);
      auto width = data_count * load->dtype.bytes();
      return Evaluate(
          Call(DataType::Void(), builtin::call_extern(),
               {StringImm("DmaDirect"), dest_addr, src_addr, width, width, width, width}));
    }
  }

  return StmtExprMutator::VisitStmt_(op);
}

namespace transform {

// Replace the whole attribute statement node whose key is "pragma_compass_dma_copy" to external
// call of Compass C low level DMA interfaces.
Pass InjectDma(Target target) {
  auto pass_func = [target](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto info = CompassInfo::Get(target);
    if (info->version == "X1") {
      n->body = DmaInjector()(std::move(n->body));
    } else {
      n->body = DmaInjectorV2()(std::move(n->body));
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectDma", {});
}

TVM_FFI_REGISTER_GLOBAL("compass.tir.transform.InjectDma").set_body_typed(InjectDma);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
