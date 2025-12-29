// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/src/runtime/packed_func_module.cc
 */
#include <tvm/runtime/module.h>

namespace tvm {
namespace runtime {

struct PackedFuncModule : public ModuleNode {
  // Internal supporting.
  // TVM framework relevant begin.
  const char* type_key() const final { return "compass.runtime.PackedFuncModule"; }
  int GetPropertyMask() const final { return ModulePropertyMask::kRunnable; }
  ffi::Function GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final;
  // TVM framework relevant end.

  std::string func_name_;  // The name of original function which generate this module.
  ffi::Function callee_;   // The packed function should be invoked.
};

ffi::Function PackedFuncModule::GetFunction(const String& name,
                                            const ObjectPtr<Object>& sptr_to_self) {
  if (name != func_name_) return nullptr;
  return callee_;
}

TVM_FFI_REGISTER_GLOBAL("compass.runtime.PackedFuncModule")
    .set_body_typed([](std::string func_name, ffi::Function callee) -> Module {
      ICHECK(callee != nullptr) << "The packed function would be invoked can't be empty.";
      ObjectPtr<PackedFuncModule> obj = make_object<PackedFuncModule>();
      obj->func_name_ = func_name;
      obj->callee_ = callee;
      return Module(obj);
    });

}  // namespace runtime
}  // namespace tvm
