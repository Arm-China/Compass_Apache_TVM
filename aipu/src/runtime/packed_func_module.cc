// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/src/runtime/packed_func_module.cc
 */
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

class PackedFuncModuleNode : public ModuleNode {
  // Things that will interface with user directly.
 public:
  explicit PackedFuncModuleNode(std::string func_name, PackedFunc callee);

  // Internal supporting.
  // Override methods that inherited from ModuleNode.
 public:
  const char* type_key() const final;

 private:
  // TVM runtime execution mechanism relevant.
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final;

  // Things of current class.
 private:
  // The name of original function which generate the current runtime module.
  std::string func_name_;
  // The packed function should be invoked.
  PackedFunc callee_;
};

const char* PackedFuncModuleNode::type_key() const { return "PackedFuncModule"; }

PackedFuncModuleNode::PackedFuncModuleNode(std::string func_name, PackedFunc callee)
    : func_name_(func_name), callee_(callee) {
  // The packed function would be invoked can't be empty.
  ICHECK(callee_ != nullptr);
  return;
}

PackedFunc PackedFuncModuleNode::GetFunction(const String& name,
                                             const ObjectPtr<Object>& sptr_to_self) {
  if (name != func_name_) return nullptr;
  return callee_;
}

Module PackedFuncModuleCreate(std::string func_name, PackedFunc callee) {
  return Module(make_object<PackedFuncModuleNode>(func_name, callee));
}

TVM_REGISTER_GLOBAL("runtime.PackedFuncModuleCreate").set_body_typed(PackedFuncModuleCreate);

}  // namespace runtime
}  // namespace tvm
