// This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
// See the copyright file distributed with this work for additional information
// regarding copyright ownership.
/*!
 * \file aipuexpr.cc
 */

#include <tvm/runtime/registry.h>
#include <tvm/tir/aipuexpr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../transforms/ir_utils.h"

namespace tvm {
namespace tir {

// Xtl
Xtl::Xtl(PrimExpr value, Span span, const std::string& predicate) {
  ICHECK(value.defined());
  ObjectPtr<XtlNode> node = make_object<XtlNode>();
  node->dtype = value.dtype().with_bits(value.dtype().bits() * 2);
  node->value = std::move(value);
  node->predicate = StrToPred(predicate);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("aipu.Xtl")
    .set_body_typed([](PrimExpr value, Span span, const std::string& predicate) {
      return Xtl(value, span, predicate);
    });

TVM_REGISTER_NODE_TYPE(XtlNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<XtlNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const XtlNode*>(node.get());
      p->stream << op->dtype << '(';
      p->Print(op->value);
      p->stream << ')';
    });

// AIPUMul
AIPUMul::AIPUMul(PrimExpr a, PrimExpr b, Span span, const std::string& predicate) {
  ICHECK(a.defined()) << "ValueError: a is undefined\n";
  ICHECK(b.defined()) << "ValueError: b is undefined\n";
  ObjectPtr<AIPUMulNode> node = make_object<AIPUMulNode>();
  node->dtype = a.dtype().with_bits(a.dtype().bits() * 2);
  node->a = std::move(a);
  node->b = std::move(b);
  node->predicate = StrToPred(predicate);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("aipu.Mul")
    .set_body_typed([](PrimExpr a, PrimExpr b, Span span, const std::string& predicate) {
      return AIPUMul(a, b, span, predicate);
    });

TVM_REGISTER_NODE_TYPE(AIPUMulNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AIPUMulNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AIPUMulNode*>(node.get());
      p->stream << '(';
      p->Print(op->a);
      p->stream << "*";
      p->Print(op->b);
      p->stream << ')';
    });

// NSRsr
NSRsr::NSRsr(DataType dtype, PrimExpr value, PrimExpr shift, bool saturation, bool round, Span span,
             const std::string& predicate) {
  ICHECK(value.defined());
  ICHECK(shift.defined());
  ObjectPtr<NSRsrNode> node = make_object<NSRsrNode>();
  node->dtype = std::move(dtype);
  node->value = std::move(value);
  node->shift = std::move(shift);
  node->saturation = saturation;
  node->round = round;
  node->predicate = StrToPred(predicate);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("aipu.NSRsr")
    .set_body_typed([](DataType dtype, PrimExpr value, PrimExpr shift, Bool saturation, Bool round,
                       Span span, const std::string& predicate) {
      return NSRsr(dtype, value, shift, saturation, round, span, predicate);
    });

TVM_REGISTER_NODE_TYPE(NSRsrNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<NSRsrNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const NSRsrNode*>(node.get());
      p->stream << op->dtype << '(';
      p->Print(op->value);
      p->stream << ",";
      p->Print(op->shift);
      p->stream << "," << op->saturation << "," << op->round << ')';
    });

// Compt
Compt::Compt(PrimExpr value, Span span, const std::string& predicate) {
  ICHECK(value.defined());
  ObjectPtr<ComptNode> node = make_object<ComptNode>();
  node->dtype = value.dtype();
  node->value = std::move(value);
  node->predicate = StrToPred(predicate);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("aipu.Compt")
    .set_body_typed([](PrimExpr value, Span span, const std::string& predicate) {
      return Compt(value, span, predicate);
    });

TVM_REGISTER_NODE_TYPE(ComptNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ComptNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ComptNode*>(node.get());
      p->stream << '(';
      p->Print(op->value);
      if (!is_one(op->predicate)) {
        p->stream << " if ";
        p->Print(op->predicate);
      }
      p->stream << ')';
    });

}  // namespace tir
}  // namespace tvm
