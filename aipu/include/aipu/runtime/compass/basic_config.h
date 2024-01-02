// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023 Arm Technology (China) Co. Ltd.
/*!
 * \file aipu/include/aipu/runtime/compass/basic_config.h
 */
#ifndef AIPU_RUNTIME_COMPASS_BASIC_CONFIG_H_
#define AIPU_RUNTIME_COMPASS_BASIC_CONFIG_H_

#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/string.h>

namespace tvm {
namespace runtime {

class AipuCompassBasicConfigObj : public Object {
  // Things that will interface with user directly.
 public:
  Map<String, String> common;
  Map<String, String> runtime;

  // Internal supporting.
  // Override things that inherited from Object.
 public:
  // TVM C++ object protocol relevant.
  static constexpr const char* _type_key = "aipu_compass.AipuCompassBasicConfig";
  static constexpr bool _type_has_method_visit_attrs = false;
  TVM_DECLARE_BASE_OBJECT_INFO(AipuCompassBasicConfigObj, Object);
};

class AipuCompassBasicConfig : public ObjectRef {
  // Things that will interface with user directly.
 public:
  String GetRuntimeWorkDir(String rly_func_name);
  static AipuCompassBasicConfig Global();
  Map<String, String> GetCommon();
  Map<String, String> GetRuntime();
  template <typename T>
  static void SetSingleton(ObjectPtr<T> obj);

  // Internal supporting.
  // Override things that inherited from ObjectRef.
 public:
  // TVM C++ object protocol relevant.
  TVM_DEFINE_OBJECT_REF_METHODS(AipuCompassBasicConfig, ObjectRef, AipuCompassBasicConfigObj);
  // Things of current class.
 private:
  static AipuCompassBasicConfig inst_;
};

template <typename T>
void AipuCompassBasicConfig::SetSingleton(ObjectPtr<T> obj) {
  static_assert(std::is_base_of<AipuCompassBasicConfigObj, T>::value,
                "Can only be set to instance of class \"AipuCompassBasicConfigObj\" or its derived "
                "classes.");

  ICHECK(obj != nullptr);
  inst_ = AipuCompassBasicConfig(obj);
  return;
}

void ConfigAipuCompass(String output_dir = "", String log_level = "0", String verbose = "false");

}  // namespace runtime
}  // namespace tvm
#endif  // AIPU_RUNTIME_COMPASS_BASIC_CONFIG_H_
