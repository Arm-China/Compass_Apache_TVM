// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/src/target/target_kind.cc
 * \brief Target kind registry
 */
#include <aipu/target/target_info.h>
#include <tvm/ir/expr.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

namespace tvm {

/*!
 * \brief Process the attributes in the AIPU target.
 * \param target The Target will be updated
 * \return The updated target
 */
TargetJSON ProcessAIPUAttrs(TargetJSON target) {
  // Check Part.
  // Check the value of the attribute "mcpu".
  if (target.count("mcpu")) {
    Array<String> valid_configs = AipuInfo::GetValidConfigNames();
    auto cfg_name = Downcast<String>(target.at("mcpu"));
    if (std::find(valid_configs.begin(), valid_configs.end(), cfg_name) == valid_configs.end()) {
      std::ostringstream oss;
      oss << ": The value of attribute \"mcpu\" of AIPU target \"" << cfg_name
          << "\" is invalid, valid choices: " << valid_configs;
      throw dmlc::Error(oss.str());
    }
  }
  return target;
}

TVM_REGISTER_TARGET_KIND("aipu", kDLAIPU)
    .add_attr_option<String>("mcpu", String("X2_1204"))
    .add_attr_option<Bool>("system-lib")
    .set_default_keys({"aipu"})
    .set_target_parser(ProcessAIPUAttrs);

}  // namespace tvm
