// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023-2025 Arm Technology (China) Co. Ltd.
/*!
 * \file compass/src/runtime/basic_config.cc
 */
#include <compass/tvm/runtime/basic_config.h>
#include <compass/tvm/runtime/utils.h>
#include <tvm/ffi/function.h>

namespace tvm {
namespace runtime {

// This statement is necessary, otherwise front-end languages can't associate
// their corresponding classes with it through "tvm.register_object".
TVM_REGISTER_OBJECT_TYPE(CompassBasicConfigObj);

CompassBasicConfig CompassBasicConfig::inst_;

String CompassBasicConfig::GetRuntimeWorkDir(String rly_func_name) {
  return get()->common["output_dir"] + "/" + rly_func_name + "/runtime";
}

void ConfigCompass(String output_dir, String verbose) {
  if (output_dir == "") {
    output_dir = "compass_output";
    if (std::optional<ffi::Function> fn = ffi::Function::GetGlobal("uuid.uuid4().hex")) {
      output_dir = output_dir + "_" + (*fn)().cast<std::string>();
    }
    output_dir = AbsPath(output_dir);
  }

  ObjectPtr<CompassBasicConfigObj> obj = make_object<CompassBasicConfigObj>();
  obj->common = {{"output_dir", output_dir}};
  obj->runtime = {{"verbose", verbose}};
  CompassBasicConfig::SetSingleton(obj);
  return;
}

TVM_FFI_REGISTER_GLOBAL("compass.runtime.ConfigCompass")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      if (args.size() == 0) {
        ConfigCompass();
      } else if (args.size() == 1) {
        ConfigCompass(args[0].cast<String>());
      } else {
        ConfigCompass(args[0].cast<String>(), args[1].cast<String>());
      }
      return;
    });

CompassBasicConfig CompassBasicConfig::Global() {
  if (!inst_.defined()) ConfigCompass();
  return inst_;
}

Map<String, String> CompassBasicConfig::GetCommon() { return inst_->common; }

Map<String, String> CompassBasicConfig::GetRuntime() { return inst_->runtime; }

TVM_FFI_REGISTER_GLOBAL("compass.runtime.CompassBasicConfig_GetCommon")
    .set_body_method(&runtime::CompassBasicConfig::GetCommon);

TVM_FFI_REGISTER_GLOBAL("compass.runtime.CompassBasicConfig_GetRuntime")
    .set_body_method(&runtime::CompassBasicConfig::GetRuntime);

TVM_FFI_REGISTER_GLOBAL("compass.runtime.CompassBasicConfig_Global")
    .set_body_typed(runtime::CompassBasicConfig::Global);

}  // namespace runtime
}  // namespace tvm
