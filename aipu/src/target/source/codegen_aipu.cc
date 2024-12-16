// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/src/target/source/codegen_aipu.cc
 */
#include "codegen_aipu_v2.h"

namespace tvm {
namespace codegen {

TVM_REGISTER_GLOBAL("target.AttachCompassModuleToLLVM").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module m = args[0];
  String target = args[1];
  const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.LLVMModuleCreate");
  runtime::Module llvm_mod = (*pf)(target, "empty_module");
  llvm_mod->Import(m);
  *rv = llvm_mod;
});

TVM_REGISTER_GLOBAL("aipu.tir.CodeGen").set_body_typed([](IRModule mod, AipuInfo aipu_info) {
  std::shared_ptr<CodeGenC> cg;
  cg = std::make_shared<CodeGenAipuV2>(aipu_info);
  cg->Init(false);

  std::vector<GlobalVar> sorted_vars(mod->functions.size());
  std::transform(mod->functions.begin(), mod->functions.end(), sorted_vars.begin(),
                 [](auto&& it) -> GlobalVar { return it.first; });
  std::sort(sorted_vars.begin(), sorted_vars.end(), [](const GlobalVar& lhs, const GlobalVar& rhs) {
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
