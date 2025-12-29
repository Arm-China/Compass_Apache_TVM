// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/src/target/codegen.cc
 */
#include "codegen_v2.h"

namespace tvm {
namespace codegen {

TVM_FFI_REGISTER_GLOBAL("target.AttachCompassModuleToLLVM")
    .set_body_typed([](runtime::Module mod, String target) -> runtime::Module {
      const ffi::Function fn = ffi::Function::GetGlobalRequired("codegen.LLVMModuleCreate");
      auto llvm_mod = fn(target, "empty_module").cast<runtime::Module>();
      llvm_mod->Import(mod);
      return llvm_mod;
    });

TVM_FFI_REGISTER_GLOBAL("compass.tir.CodeGen")
    .set_body_typed([](IRModule mod, CompassInfo cps_info) {
      std::shared_ptr<CodeGenC> cg;
      cg = std::make_shared<CodeGenCompassV2>(cps_info);
      cg->Init(false);

      std::vector<GlobalVar> sorted_vars(mod->functions.size());
      std::transform(mod->functions.begin(), mod->functions.end(), sorted_vars.begin(),
                     [](auto&& it) -> GlobalVar { return it.first; });
      std::sort(sorted_vars.begin(), sorted_vars.end(),
                [](const GlobalVar& lhs, const GlobalVar& rhs) {
                  return lhs.as<GlobalVarNode>()->name_hint < rhs.as<GlobalVarNode>()->name_hint;
                });

      for (auto var : sorted_vars) {
        auto prim_func = Downcast<PrimFunc>(mod->functions.at(var));
        try {
          cg->DeclareFunction(var, prim_func);
        } catch (const dmlc::Error& e) {
          std::ostringstream oss;
          oss << e.what() << prim_func;
          throw dmlc::Error(oss.str());
        }
      }

      for (auto var : sorted_vars) {
        auto prim_func = Downcast<PrimFunc>(mod->functions.at(var));
        try {
          cg->AddFunction(var, prim_func);
        } catch (const dmlc::Error& e) {
          std::ostringstream oss;
          oss << e.what() << prim_func;
          throw dmlc::Error(oss.str());
        }
      }

      return cg->Finish();
    });

}  // namespace codegen
}  // namespace tvm
