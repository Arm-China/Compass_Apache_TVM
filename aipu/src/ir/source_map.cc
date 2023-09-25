// This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
// See the copyright file distributed with this work for additional information
// regarding copyright ownership.
/*!
 * \file aipu/src/ir/source_map.cc
 */
#include <tvm/ir/source_map.h>
#include <tvm/ir/transform.h>

namespace tvm {

TVM_REGISTER_PASS_CONFIG_OPTION("relay.frontend.span_propagation", Bool);

}  // namespace tvm
