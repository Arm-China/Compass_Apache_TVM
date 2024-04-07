// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/src/ir/source_map.cc
 */
#include <tvm/ir/source_map.h>
#include <tvm/ir/transform.h>

namespace tvm {

TVM_REGISTER_PASS_CONFIG_OPTION("relay.frontend.span_propagation", Bool);

}  // namespace tvm
