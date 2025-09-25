// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file nms.h
 * \brief The functions to make Relax Non-maximum suppression operator calls.
 */

#ifndef TVM_RELAX_OP_VISION_NMS_H_
#define TVM_RELAX_OP_VISION_NMS_H_

#include <tvm/relax/attrs/vision.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*! \brief Compute All Class NonMaximumSuppression. */
Expr all_class_non_max_suppression(Expr boxes, Expr scores, Expr max_output_boxes_per_class,
                                   Expr iou_threshold, Expr score_threshold, String output_format);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_VISION_NMS_H_
