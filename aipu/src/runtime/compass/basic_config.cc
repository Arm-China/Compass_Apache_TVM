// This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
// See the copyright file distributed with this work for additional information
// regarding copyright ownership.
/*!
 * \file aipu/src/runtime/compass/basic_config.cc
 */
#include <aipu/runtime/compass/basic_config.h>
#include <aipu/runtime/utils.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

// This statement is necessary, otherwise front-end languages can't associate
// their corresponding classes with it through "tvm.register_object".
TVM_REGISTER_OBJECT_TYPE(AipuCompassBasicConfigObj);

AipuCompassBasicConfig AipuCompassBasicConfig::inst_;

String AipuCompassBasicConfig::GetRuntimeWorkDir(String rly_func_name) {
  return get()->common["output_dir"] + "/" + rly_func_name + "/runtime";
}

void ConfigAipuCompass(String output_dir, String log_level, String verbose) {
  if (output_dir == "") {
    output_dir = "compass_output";
    if (const auto* f = Registry::Get("uuid.uuid4().hex")) {
      output_dir = output_dir + "_" + (*f)().operator std::string();
    }
    output_dir = AbsPath(output_dir);
  }

  ObjectPtr<AipuCompassBasicConfigObj> obj = make_object<AipuCompassBasicConfigObj>();
  obj->common = {{"output_dir", output_dir}};
  obj->runtime = {{"log_level", log_level}, {"verbose", verbose}};
  AipuCompassBasicConfig::SetSingleton(obj);
  return;
}

TVM_REGISTER_GLOBAL("aipu_compass.ConfigAipuCompass").set_body([](TVMArgs args, TVMRetValue* rv) {
  if (args.num_args == 0) {
    ConfigAipuCompass();
  } else if (args.num_args == 1) {
    ConfigAipuCompass(args[0]);
  } else if (args.num_args == 2) {
    ConfigAipuCompass(args[0], args[1]);
  } else {
    ConfigAipuCompass(args[0], args[1], args[2]);
  }
  return;
});

AipuCompassBasicConfig AipuCompassBasicConfig::Global() {
  if (!inst_.defined()) ConfigAipuCompass();
  return inst_;
}

}  // namespace runtime
}  // namespace tvm
