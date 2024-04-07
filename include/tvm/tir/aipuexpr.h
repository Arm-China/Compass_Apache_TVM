// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file aipuexpr.h
 */

#ifndef TVM_TIR_AIPUEXPR_H_
#define TVM_TIR_AIPUEXPR_H_

#include <tvm/ir/expr.h>

#include <string>

namespace tvm {
namespace tir {

/**
 * \brief Base class to implement AIPU ops.
 */
class BaseAIPUOpNode : public PrimExprNode {
 public:
  /*! \brief The predicate to mask which lanes would be operate. */
  Array<Bool> predicate;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &(this->dtype));
    v->Visit("predicate", &predicate);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const BaseAIPUOpNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(predicate, other->predicate);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(predicate);
  }
};

/*!
 * \brief Extend from low bits to high bits.
 */
class XtlNode : public BaseAIPUOpNode {
 public:
  /*! \brief Original data. */
  PrimExpr value;

  void VisitAttrs(AttrVisitor* v) {
    BaseAIPUOpNode::VisitAttrs(v);
    v->Visit("value", &value);
  }

  bool SEqualReduce(const XtlNode* other, SEqualReducer equal) const {
    return equal(value, other->value) && BaseAIPUOpNode::SEqualReduce(other, equal);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    BaseAIPUOpNode::SHashReduce(hash_reduce);
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "aipu.Xtl";
  TVM_DECLARE_FINAL_OBJECT_INFO(XtlNode, PrimExprNode);
};

/*!
 * \brief Managed reference to XtlNode
 * \sa XtlNode
 */
class Xtl : public PrimExpr {
 public:
  TVM_DLL Xtl(PrimExpr value, Span span = Span(), const std::string& predicate = "0xffffffff");
  TVM_DEFINE_OBJECT_REF_METHODS(Xtl, PrimExpr, XtlNode);
};

/*!
 * \brief AIPU mul node.
 */
class AIPUMulNode : public BaseAIPUOpNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;

  void VisitAttrs(AttrVisitor* v) {
    BaseAIPUOpNode::VisitAttrs(v);
    v->Visit("a", &a);
    v->Visit("b", &b);
  }

  bool SEqualReduce(const AIPUMulNode* other, SEqualReducer equal) const {
    return equal(a, other->a) && equal(b, other->b) && BaseAIPUOpNode::SEqualReduce(other, equal);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    BaseAIPUOpNode::SHashReduce(hash_reduce);
    hash_reduce(a);
    hash_reduce(b);
  }

  static constexpr const char* _type_key = "aipu.Mul";
  TVM_DECLARE_FINAL_OBJECT_INFO(AIPUMulNode, PrimExprNode);
};

/*!
 * \brief Managed reference to AIPUMulNode
 * \sa AIPUMulNode
 */
class AIPUMul : public PrimExpr {
 public:
  TVM_DLL AIPUMul(PrimExpr a, PrimExpr b, Span span = Span(),
                  const std::string& predicate = "0xffffffff");
  TVM_DEFINE_OBJECT_REF_METHODS(AIPUMul, PrimExpr, AIPUMulNode);
};

/*!
 * \brief Narrow shift right with optional saturation and round.
 */
class NSRsrNode : public BaseAIPUOpNode {
 public:
  /*! \brief Left operand. */
  PrimExpr value;
  /*! \brief Right operand. */
  PrimExpr shift;
  /*! \brief Whether saturation. */
  bool saturation;
  /*! \brief Whether round. */
  bool round;

  void VisitAttrs(AttrVisitor* v) {
    BaseAIPUOpNode::VisitAttrs(v);
    v->Visit("value", &value);
    v->Visit("shift", &shift);
    v->Visit("saturation", &saturation);
    v->Visit("round", &round);
  }

  bool SEqualReduce(const NSRsrNode* other, SEqualReducer equal) const {
    return equal(value, other->value) && equal(shift, other->shift) &&
           equal(saturation, other->saturation) && equal(round, other->round) &&
           BaseAIPUOpNode::SEqualReduce(other, equal);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    BaseAIPUOpNode::SHashReduce(hash_reduce);
    hash_reduce(value);
    hash_reduce(shift);
    hash_reduce(saturation);
    hash_reduce(round);
  }

  static constexpr const char* _type_key = "aipu.NSRsr";
  TVM_DECLARE_FINAL_OBJECT_INFO(NSRsrNode, PrimExprNode);
};

/*!
 * \brief Managed reference to NSRsrNode
 * \sa NSRsrNode
 */
class NSRsr : public PrimExpr {
 public:
  TVM_DLL NSRsr(DataType dtype, PrimExpr value, PrimExpr shift, bool saturation = false,
                bool round = false, Span span = Span(),
                const std::string& predicate = "0xffffffff");
  TVM_DEFINE_OBJECT_REF_METHODS(NSRsr, PrimExpr, NSRsrNode);
};

/*!
 * \brief Packs selected elements into the lowest-numbered elements.
 */
class ComptNode : public BaseAIPUOpNode {
 public:
  /*! \brief Original data. */
  PrimExpr value;

  void VisitAttrs(AttrVisitor* v) {
    BaseAIPUOpNode::VisitAttrs(v);
    v->Visit("value", &value);
  }

  bool SEqualReduce(const ComptNode* other, SEqualReducer equal) const {
    return equal(value, other->value) && BaseAIPUOpNode::SEqualReduce(other, equal);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    BaseAIPUOpNode::SHashReduce(hash_reduce);
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "aipu.Compt";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComptNode, PrimExprNode);
};

/*!
 * \brief Managed reference to ComptNode
 * \sa ComptNode
 */
class Compt : public PrimExpr {
 public:
  TVM_DLL Compt(PrimExpr value, Span span = Span(), const std::string& predicate = "0xffffffff");
  TVM_DEFINE_OBJECT_REF_METHODS(Compt, PrimExpr, ComptNode);
};

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_AIPUEXPR_H_
