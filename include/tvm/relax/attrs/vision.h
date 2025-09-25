// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file tvm/relax/attrs/vision.h
 * \brief Auxiliary attributes for vision operators.
 */
#ifndef TVM_RELAX_ATTRS_VISION_H_
#define TVM_RELAX_ATTRS_VISION_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in AllClassNonMaximumSuppression operator */
struct AllClassNonMaximumSuppressionAttrs
    : public tvm::AttrsNode<AllClassNonMaximumSuppressionAttrs> {
  String output_format;

  TVM_DECLARE_ATTRS(AllClassNonMaximumSuppressionAttrs,
                    "relax.attrs.AllClassNonMaximumSuppressionAttrs") {
    TVM_ATTR_FIELD(output_format)
        .set_default("onnx")
        .describe(
            "Output format, onnx or tensorflow. Returns outputs in a way that can be easily "
            "consumed by each frontend.");
  }
};  // struct AllClassNonMaximumSuppressionAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_VISION_H_
