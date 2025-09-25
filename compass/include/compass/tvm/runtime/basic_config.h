// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/include/compass/tvm/runtime/basic_config.h
 */
#ifndef COMPASS_TVM_RUNTIME_BASIC_CONFIG_H_
#define COMPASS_TVM_RUNTIME_BASIC_CONFIG_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/object.h>

namespace tvm {
namespace runtime {

class CompassBasicConfigObj : public Object {
  // Things that will interface with user directly.
 public:
  Map<String, String> common;
  Map<String, String> runtime;

  // Internal supporting.
  // Override things that inherited from Object.
 public:
  // TVM C++ object protocol relevant.
  static constexpr const char* _type_key = "compass.runtime.CompassBasicConfig";
  static constexpr const bool _type_has_method_visit_attrs = false;
  TVM_DECLARE_BASE_OBJECT_INFO(CompassBasicConfigObj, Object);
};

class CompassBasicConfig : public ObjectRef {
  // Things that will interface with user directly.
 public:
  String GetRuntimeWorkDir(String rly_func_name);
  static CompassBasicConfig Global();
  Map<String, String> GetCommon();
  Map<String, String> GetRuntime();
  template <typename T>
  static void SetSingleton(ObjectPtr<T> obj);

  // Internal supporting.
  // Override things that inherited from ObjectRef.
 public:
  // TVM C++ object protocol relevant.
  TVM_DEFINE_OBJECT_REF_METHODS(CompassBasicConfig, ObjectRef, CompassBasicConfigObj);
  // Things of current class.
 private:
  static CompassBasicConfig inst_;
};

template <typename T>
void CompassBasicConfig::SetSingleton(ObjectPtr<T> obj) {
  static_assert(std::is_base_of<CompassBasicConfigObj, T>::value,
                "Can only be set to instance of class \"CompassBasicConfigObj\" or its derived "
                "classes.");

  ICHECK(obj != nullptr);
  inst_ = CompassBasicConfig(obj);
  return;
}

void ConfigCompass(String output_dir = "", String verbose = "false");

}  // namespace runtime
}  // namespace tvm
#endif  // COMPASS_TVM_RUNTIME_BASIC_CONFIG_H_
